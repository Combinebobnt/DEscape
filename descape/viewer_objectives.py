"""ViewerWindow's Objectives dock (GH #391): the panel's refresh, its writes,
selection sync with the Triggers panel and the on-map preview.

Every write goes through _trigger_edit(), one undo record per action, with
unchanged values filtered out first. New Objective and New Header add a
trigger through structural_edit() in the same record that sets its flags.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import QApplication, QMessageBox

from descape import settings, trigger_organize
from descape.objectives_model import has_string_id, insert_orders, objective_rows
from descape.objectives_panel import NO_DOCUMENT, NO_TRIGGERS, PLACE_DOCK, PLACE_MAP
from descape.scenario_io import parse_triggers
from descape.trigger_model import display_order_with_block_inserted
from descape.trigger_status import _OBJECTIVE_FLAGS, read_value

# The library's (and a new trigger's) "no string id".
NO_STRING_ID = -1
READ_ONLY_NOTE = "Triggers are read-only for this file: the list and preview still work."
_NEW_NAMES = {"objective": "New objective", "header": "New header"}


def _differs(current, value) -> bool:
    if isinstance(value, bool):
        return bool(current) != value
    return current != value


class _ViewerObjectivesMixin:
    """ViewerWindow's Objectives dock (viewer.py's docstring has the split's rules)."""

    def refresh_objectives(self) -> None:
        """Re-read the rows (never cached, as trigger_panel.show_scenario()
        says). Only in Triggers mode, the dock's only mode: entering it
        refreshes, and another document clears it until then. A new document,
        or a trigger added, deleted or reordered, drops the preview ticks."""
        panel = self.objectives_panel
        if self.scenario is None:
            self._objectives_doc = None
            panel.clear_document(NO_DOCUMENT)
        elif self.mode != "triggers":
            if self._objectives_doc is None or self._objectives_doc[0] != self._doc_id:
                self._objectives_doc = None
                panel.clear_document(NO_DOCUMENT)
        else:
            manager = parse_triggers(self.scenario)
            # By trigger object, not count: a reorder at the same count (Renumber IDs) moves ticks onto other triggers.
            key = (self._doc_id, None if manager is None else tuple(map(id, manager.triggers)))
            if key != self._objectives_doc:
                self._objectives_doc = key
                panel.clear_preview()
            if manager is None:
                panel.clear_document(NO_TRIGGERS)
            else:
                reason = self.game_data_lock_reason()
                writable = bool(self.scenario.trigger_write_supported)
                note = reason or ("" if writable else READ_ONLY_NOTE)
                panel.set_variables([v.name or "" for v in manager.variables])
                panel.show_rows(objective_rows(manager), writable and reason is None, note)
                self._objectives_follow_trigger()
        self._push_objectives_overlay()

    def _objective_write(self, label: str, changes: dict[int, dict[str, object]]) -> None:
        """One recorded edit setting each trigger's attributes; nothing when no value changes."""
        model = self._ensure_trigger_edits()
        if model is None:
            self.refresh_objectives()
            return
        triggers = model.manager().triggers
        changed = {}
        for index, attrs in changes.items():
            if 0 <= index < len(triggers):
                differing = {a: v for a, v in attrs.items() if _differs(read_value(triggers[index], a), v)}
                if differing:
                    changed[index] = differing
        if not changed:
            self.refresh_objectives()
            return
        # "panel" repopulates the Triggers panel, whose _show_triggers() refreshes this list too.
        with self._trigger_edit(model, label, content_touched=sorted(changed), refresh="panel") as m:
            live = m.manager().triggers
            for index, attrs in changed.items():
                for attr, value in attrs.items():
                    setattr(live[index], attr, int(value) if isinstance(value, bool) else value)

    def _objective_flags(self, indices: Sequence[int], attr: str, value: bool) -> None:
        self._objective_write("Set objective flags", {index: {attr: bool(value)} for index in indices})

    def _objective_order(self, changes: dict[int, int], label: str = "Reorder objectives") -> None:
        self._objective_write(label, {index: {"description_order": int(order)} for index, order in changes.items()})

    def _objective_short(self, index: int, text: str) -> None:
        """An inline edit. A row the game shows a language string for asks
        whether to clear that id too, so the typed text is what shows."""
        attrs: dict[str, object] = {"short_description": text}
        model = self._ensure_trigger_edits()
        if model is None:
            return
        triggers = model.manager().triggers
        stid = read_value(triggers[index], "short_description_stid") if 0 <= index < len(triggers) else None
        if isinstance(stid, int) and has_string_id(stid):
            answer = QMessageBox.question(
                self,
                "Objective shows a language string",
                f"The game shows language string #{stid} for this objective, not its text.\n\n"
                "Clear the string id so your text shows? Keeping it saves the text, "
                "which the game shows only when that string is missing.",
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
                QMessageBox.Yes,
            )
            if answer == QMessageBox.Cancel:
                self.refresh_objectives()
                return
            if answer == QMessageBox.Yes:
                attrs["short_description_stid"] = NO_STRING_ID
        self._objective_write("Set objective text", {index: attrs})

    def _objective_clear_string_ids(self, indices: Sequence[int]) -> None:
        self._objective_write("Use objective text", {index: {"short_description_stid": NO_STRING_ID} for index in indices})

    def _objective_remove(self, indices: Sequence[int]) -> None:
        self._objective_write("Clear objective", {index: dict.fromkeys(_OBJECTIVE_FLAGS, False) for index in indices})

    def _objective_promote(self, indices: Sequence[int]) -> None:
        """Add Trigger…: Display as Objective and on Screen set, placed under the selected row."""
        model = self._ensure_trigger_edits()
        if model is None:
            return
        rows = objective_rows(model.manager())
        present = {row.index for row in rows}
        new = [index for index in dict.fromkeys(indices) if index not in present]
        orders = insert_orders(rows, self.objectives_panel.current_index(), new)
        changes = {index: {"description_order": order} for index, order in orders.items()}
        for index in new:
            changes[index].update(display_as_objective=True, display_on_screen=True)
        self._objective_write("Make objective", changes)
        if new:
            self.trigger_panel.select_triggers(new)
            self.objectives_panel.select_index(new[0])
            self.objectives_panel.select_indices(new)

    def _objective_new(self, kind: str, after_index: int | None) -> None:
        """New Objective or New Header: one record that adds the trigger under
        `after_index`'s trigger in display order, with its flags, text and
        an order between its new neighbours, then shows it in the Triggers panel."""
        model = self._ensure_trigger_edits()
        if model is None:
            return
        manager = model.manager()
        rows = objective_rows(manager)
        new_index = len(manager.triggers)
        orders = insert_orders(rows, after_index, [new_index])
        renumbered = {index: order for index, order in orders.items() if index != new_index}
        header = kind == "header"
        taken = {(t.name or "") for t in manager.triggers}
        base = _NEW_NAMES[kind]
        name = base if base not in taken else next(f"{base} {n}" for n in range(2, len(taken) + 3) if f"{base} {n}" not in taken)
        anchor = after_index if after_index is not None and 0 <= after_index < new_index else None

        def mutate(m):
            before_order = list(m.trigger_display_order)
            m.add_trigger(
                name,
                short_description=name if header else f"+ {name}",
                display_as_objective=1,
                display_on_screen=1,
                header=int(header),
                description_order=orders[new_index],
            )
            for index, order in renumbered.items():
                m.triggers[index].description_order = order
            m.trigger_display_order = display_order_with_block_inserted(before_order, [new_index], anchor)

        label = "New objective header" if header else "New objective"
        with self._trigger_edit(model, label, content_touched=sorted(renumbered), refresh="panel") as m:
            m.structural_edit(mutate, content_touched=sorted(renumbered))
        if len(model.manager().triggers) > new_index:
            self.trigger_panel.select_triggers([new_index])
            self.trigger_panel.reveal_trigger(new_index)
        self.objectives_panel.select_index(new_index)

    def _objective_candidates(self) -> list[tuple[int, str]]:
        """Add Trigger…'s list: every trigger that is not an objective, header,
        on-screen line or section divider."""
        if self.scenario is None or (manager := parse_triggers(self.scenario)) is None:
            return []
        return [
            (index, f"#{read_value(trigger, 'trigger_id')} {trigger.name or ''}")
            for index, trigger in enumerate(manager.triggers)
            if not any(read_value(trigger, flag) for flag in _OBJECTIVE_FLAGS)
            and not trigger_organize.is_divider(trigger.name or "")
        ]

    def _objective_select(self, index: int) -> None:
        """A row clicked: its trigger becomes the Triggers panel's selection.
        The panel's filter is cleared only when it hides that trigger."""
        if self.scenario is None:
            return
        self._objectives_selecting = True
        try:
            self.trigger_panel.select_triggers([index])
            if index not in self.trigger_panel.selected_trigger_indices():
                self._reveal_trigger_entry(index, "", -1)
                self.trigger_panel.select_triggers([index])
        finally:
            self._objectives_selecting = False

    def _objectives_list_focused(self) -> bool:
        """Whether keys go to the Objectives list, so the Edit menu's commands act there, not on the triggers."""
        focus = QApplication.focusWidget()
        tree = self.objectives_panel.tree
        return focus is not None and (focus is tree or tree.isAncestorOf(focus))

    def _objectives_follow_trigger(self) -> None:
        """The Triggers panel's current trigger highlights its row, if it has one."""
        if self._objectives_selecting or self.mode != "triggers":
            return
        self.objectives_panel.select_index(self.trigger_panel.current_trigger_index())

    # -- tab height -----------------------------------------------------------

    def _on_objectives_visibility(self, visible: bool) -> None:
        """Tabbed with the Status Log, this fires True when the Objectives tab comes to the front
        and False when it leaves it; the dock's height settles only after the switch."""
        self._objectives_front = visible
        self._objectives_later(self._fit_objectives_row)

    def _bottom_row(self) -> int:
        # The row's height as the window minus the map: steady when a mode change drops the tab bar with the dock.
        return self.height() - self.centralWidget().height()

    def _fit_objectives_row(self) -> None:
        """The default tab behind the Status Log is a few log lines tall. Brought to the front, the
        row grows to fit the list and its buttons, up to half the window; sent back (another tab,
        another mode), the row returns to its old height unless it was dragged in between."""
        if not self.isVisible():
            return
        dock = self.objectives_dock
        under_map = self.dockWidgetArea(self.log_dock) in (Qt.TopDockWidgetArea, Qt.BottomDockWidgetArea)
        tabbed = under_map and not dock.isFloating() and self.log_dock in self.tabifiedDockWidgets(dock)
        if self._objectives_front:
            if self._objectives_row is not None:
                return
            title_bar = dock.height() - self.objectives_panel.height()
            target = min(self.objectives_panel.fit_height() + title_bar, self.height() // 2)
            if tabbed and dock.height() < target:
                before = self._bottom_row()
                self._resize_row(dock, target, self._note_objectives_row(before))
            return
        grown, self._objectives_row = self._objectives_row, None
        if grown is not None and under_map and not self.log_dock.isFloating() and self._bottom_row() == grown[1]:
            self._resize_row(self.log_dock, self.log_dock.height() - (grown[1] - grown[0]))

    def _shrink_objectives_row(self) -> None:
        """Before the layout is saved: a grown row is the tab's, not the user's, so it is not kept."""
        grown, self._objectives_row = self._objectives_row, None
        if grown is not None and self._objectives_front and self._bottom_row() == grown[1]:
            dock = self.objectives_dock
            self.resizeDocks([dock], [dock.height() - (grown[1] - grown[0]) + grown[2]], Qt.Vertical)

    def _resize_row(self, dock, height: int, then=None) -> None:
        """resizeDocks() on a tabbed dock counts the tab bar, which the dock's height does not,
        so one more pass corrects by whatever the first one missed."""
        self.resizeDocks([dock], [height], Qt.Vertical)

        def correct() -> None:
            short = height - dock.height()
            if short:
                self.resizeDocks([dock], [height + short], Qt.Vertical)
            if then is not None:
                self._objectives_later(lambda: then(max(short, 0)))

        self._objectives_later(correct)

    def _note_objectives_row(self, before: int) -> Callable[[int], None]:
        def note(tab_bar: int) -> None:
            self._objectives_row = (before, self._bottom_row(), tab_bar)

        return note

    def _objectives_later(self, call) -> None:
        # A child timer, so a later call runs on this window's thread; _fit_objectives_row() skips a closed window.
        timer = QTimer(self)
        timer.setSingleShot(True)
        timer.timeout.connect(call)
        timer.timeout.connect(timer.deleteLater)
        timer.start(0)

    # -- preview --------------------------------------------------------------

    def _push_objectives_overlay(self) -> None:
        """The map's box: shown in Triggers mode with Preview On map only."""
        shown = (
            self.scenario is not None and self.mode == "triggers" and self.objectives_panel.preview_place() == PLACE_MAP
        )
        self.objectives_overlay.set_shown(shown)

    def _on_objectives_preview_place(self, place: str) -> None:
        settings.set_objectives_preview(place)
        action = self.objectives_overlay_action
        action.blockSignals(True)
        action.setChecked(place == PLACE_MAP)
        action.blockSignals(False)
        self._push_objectives_overlay()

    def _on_objectives_overlay_toggled(self, checked: bool) -> None:
        """View > Show Objectives on Map: On map, or back to In dock."""
        place = self.objectives_panel.preview_place()
        if checked == (place == PLACE_MAP):
            return
        place = PLACE_MAP if checked else PLACE_DOCK
        self.objectives_panel.set_preview_place(place)
        settings.set_objectives_preview(place)
        self._push_objectives_overlay()

    def _on_objectives_at_start(self, at_start: bool) -> None:
        settings.set_objectives_at_start(at_start)
