"""TriggerPanel's trigger list: selection preservation across repopulates, the trigger
and entry trees, the exec-order readout and the multi-select page."""

from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QTreeWidgetItem,
)

from descape import (
    trigger_fields,
)
from descape.scenario_io import (
    LoadedScenario,
)
from descape.trigger_model import (
    EXEC_MODE_LEGACY,
    EXEC_MODE_UNKNOWN,
    entry_display_order,
    exec_order_value,
    resolve_exec_mode,
)


class _TriggerPanelListMixin:
    """TriggerPanel's trigger list and selection (trigger_panel.py's docstring has the split's rules)."""

    # -- selection preservation ----------------------------------------------

    def _selection_state(self) -> tuple:
        """(current trigger path, selected trigger paths, entry paths,
        scroll) so a rebuild keeps the user's place. By list index / tree path, never by trigger_id: ids are
        positional and a reorder or removal renumbers the whole list.

        The entry element is (current entry path, the other selected entry
        paths), so a multi-entry selection (GH #60) survives a rebuild whole.
        """
        current_entry = self.entry_tree.currentItem()
        entry_path = self._entry_row_path(current_entry)
        other_entries = tuple(
            self._entry_row_path(item)
            for item in self.entry_tree.selectedItems()
            if item is not current_entry
        )
        current = self.tree.currentItem()
        selected = tuple(
            self._trigger_tree_path(item) for item in self.tree.selectedItems() if item is not current
        )
        return (
            self._trigger_tree_path(current),
            selected,
            (entry_path, other_entries),
            self.tree.verticalScrollBar().value(),
        )

    def _entry_row_path(self, item) -> tuple[int, int | None] | None:
        """(top-level row, child row) for `item` in self.entry_tree, or None;
        the same shape _trigger_tree_path() uses for the trigger tree."""
        if item is None:
            return None
        parent = item.parent()
        if parent is None:
            return (self.entry_tree.indexOfTopLevelItem(item), None)
        return (self.entry_tree.indexOfTopLevelItem(parent), parent.indexOfChild(item))

    def _entry_item_at_path(self, path):
        """The entry-tree item at an _entry_row_path() path, or None."""
        if path is None:
            return None
        top, child = path
        if not 0 <= top < self.entry_tree.topLevelItemCount():
            return None
        item = self.entry_tree.topLevelItem(top)
        if child is None:
            return item
        return item.child(child) if 0 <= child < item.childCount() else None

    def _trigger_tree_path(self, item) -> tuple[int, int] | None:
        """(top-level row, child row) for `item` in self.tree, or None.

        child row is None when `item` is itself a top-level row -- the same
        shape _selection_state() already uses for the entry tree above, so a
        flat document (no children anywhere) restores exactly as it did
        before 4c grouping existed.
        """
        if item is None:
            return None
        parent = item.parent()
        if parent is None:
            return (self.tree.indexOfTopLevelItem(item), None)
        return (self.tree.indexOfTopLevelItem(parent), parent.indexOfChild(item))

    def _restore_selection(self, state: tuple) -> None:
        trigger_path, selected_paths, (entry_path, other_entry_paths), scroll = state
        # No autoScroll: scrollTo() would expand a collapsed section holding
        # the current trigger, which the rebuild only restored (GH #135).
        auto_scroll = self.tree.hasAutoScroll()
        self.tree.setAutoScroll(False)
        try:
            self._restore_trigger_path(trigger_path)
        finally:
            self.tree.setAutoScroll(auto_scroll)
        # After the current item, never before: setCurrentItem() clears the
        # selection (ClearAndSelect).
        for path in selected_paths:
            item = self._item_at_path(path)
            if item is not None and item.data(self._ROLE_COL, Qt.UserRole) is not None:
                item.setSelected(True)
        self.tree.verticalScrollBar().setValue(scroll)
        item = self._entry_item_at_path(entry_path)
        if item is None:
            return
        self._set_current_entry_item(item)
        # The rest after the current one, for the trigger tree's reason above.
        for path in other_entry_paths:
            other = self._entry_item_at_path(path)
            if other is not None:
                other.setSelected(True)

    def _item_at_path(self, path: tuple[int, int] | None):
        """The trigger-tree item at a _trigger_tree_path() path, or None."""
        item = None
        if path is not None:
            top, child = path
            if 0 <= top < self.tree.topLevelItemCount():
                item = self.tree.topLevelItem(top)
                if child is not None and 0 <= child < item.childCount():
                    item = item.child(child)
        return item

    def _restore_trigger_path(self, path: tuple[int, int] | None) -> None:
        item = self._item_at_path(path)
        # Falls back to the default selection rather than the raw item on any
        # miss -- an out-of-range path, or a resolved item that turned out to
        # be the synthetic header (grouping can restructure which row a given
        # path now points at, e.g. after a rename crosses the divider
        # predicate -- see _changed()).
        if item is not None and item.data(self._ROLE_COL, Qt.UserRole) is not None:
            self._set_current_trigger_item(item)
        else:
            self._select_default_trigger()

    # -- the trigger list ----------------------------------------------------

    def _trigger_label(self, trigger) -> str:
        name = (self._read(trigger, "name") or "").strip() or "(unnamed)"
        return name if self._read(trigger, "enabled") else f"{name}  (disabled)"

    def _exec_order_readout(
        self, loaded: LoadedScenario, pending_exec_order: int | None
    ) -> str:
        """Read-only: which axis actually governs execution. Changing it is a
        Map Options edit, not a trigger edit, so this never touches
        `self.trigger_edits` (there is none -- the panel doesn't own one) or
        builds a TriggerEditModel -- see exec_order_value()'s docstring.
        `pending_exec_order` is the caller-resolved value of an in-progress
        edit, passed in rather than read through a model here for that same
        reason.
        """
        mode, unsaved = resolve_exec_mode(exec_order_value(loaded), pending_exec_order)
        if mode == EXEC_MODE_UNKNOWN:
            return "Execution order is not stored in this file."
        suffix = " - unsaved change." if unsaved else "."
        if mode == EXEC_MODE_LEGACY:
            return f"Executes in trigger-ID order (legacy){suffix}"
        return f"Executes in display order{suffix}"

    def current_trigger_index(self) -> int | None:
        """The trigger's stable list index, not its row -- these diverge under
        a display-order sort. Each item's Qt.UserRole carries the index set
        at populate time (show_scenario()); nothing here assumes row order.

        Also None when the selected row is 4c's one synthetic row (the
        "(before the first section)" header) -- its Qt.UserRole is None by
        construction (_populate_grouped()), which is what every consumer
        below already treats as "no trigger selected" without a special case.
        """
        item = self.tree.currentItem()
        if item is None:
            return None
        return item.data(self._ROLE_COL, Qt.UserRole)

    def selected_trigger_indices(self) -> list[int]:
        """Every selected trigger's list index, ordered by display position
        whatever the sort mode: the order a paste reproduces and Move acts in.

        Skips 4c's synthetic header (Qt.UserRole None) and any row a filter
        hides, so a filtered-out row is never swept into a bulk verb.
        """
        indices = []
        for item in self.tree.selectedItems():
            index = item.data(self._ROLE_COL, Qt.UserRole)
            if index is None or item.isHidden():
                continue
            indices.append(index)
        return sorted(indices, key=lambda index: self._display_position.get(index, index))

    def selected_block_indices(self) -> list[int]:
        """selected_trigger_indices() plus every visible member of each
        selected section header whose section is collapsed (GH #134): a
        collapsed row stands for its whole section, like a folder.

        A fourth accessor beside the selection contract's three, read by
        Duplicate, Delete and Copy/Cut (button, menu, Edit). Everything else
        (the detail page, the form, the entry tree, Move Up/Down, the counts)
        stays on the raw selection, so a collapsed header still shows its own
        form. An expanded header stays header-only.
        """
        indices = set(self.selected_trigger_indices())
        for index in list(indices):
            item = self._item_for_index.get(index)
            if item is None or item.parent() is not None or item.isExpanded():
                continue
            section = self._section_for(item)
            if section is None or section.header_index != index:
                continue
            indices.update(m for m in section.member_indices if not self._item_for_index[m].isHidden())
        return sorted(indices, key=lambda index: self._display_position.get(index, index))

    def _paste_anchor_index(self, index: int | None) -> int | None:
        """Where a paste below `index` lands (GH #134): a collapsed section
        header resolves to its section's last member, the header itself when
        it has none. Anything else is `index` unchanged."""
        item = None if index is None else self._item_for_index.get(index)
        if item is None or item.parent() is not None or item.isExpanded():
            return index
        section = self._section_for(item)
        if section is None or section.header_index != index or not section.member_indices:
            return index
        return section.member_indices[-1]

    def current_paste_anchor(self) -> int | None:
        """Where Paste (button or Edit > Paste Triggers) lands: below the current trigger, or its collapsed section's end."""
        return self._paste_anchor_index(self.current_trigger_index())

    def _on_trigger_selected(self) -> None:
        if self._populating:
            return
        if len(self.selected_trigger_indices()) > 1:
            # The selection handler owns the multi state; populating the
            # current trigger's entries here would only be cleared again.
            self._show_multi_page()
        else:
            self._populate_entry_tree()
        self._update_buttons()

    def _on_trigger_selection_changed(self) -> None:
        """The detail page and buttons follow the selection *set*. The
        entry tree is repopulated only when leaving the multi page, since
        currentItemChanged does not fire when the current item stays put."""
        if self._populating:
            return
        if len(self.selected_trigger_indices()) > 1:
            self._show_multi_page()
        elif self.detail_stack.currentIndex() == self._DETAIL_MULTI:
            self.detail_stack.setCurrentIndex(self._DETAIL_FORM)
            self._populate_entry_tree()
        self._update_buttons()
        self._on_selection_changed()

    def _show_multi_page(self) -> None:
        """The Units-mode precedent: N selected shows a count, not a form.
        The entry tree is cleared too, or it would keep showing (and its
        buttons keep acting on) the current trigger alone."""
        if self.detail_stack.currentIndex() == self._DETAIL_PICKER:
            self._close_picker()
        count = len(self.selected_trigger_indices())
        self.multi_label.setText(f"{count} triggers selected")
        self._populating = True
        try:
            self._clear_entry_tree()
            self._clear_property_form()
        finally:
            self._populating = False
        self.detail_stack.setCurrentIndex(self._DETAIL_MULTI)

    def _clear_entry_tree(self) -> None:
        # Together, or _entry_item_for() hands back a deleted row.
        self._entry_items = {}
        self.entry_tree.clear()

    def _populate_entry_tree(self) -> None:
        self._populating = True
        try:
            self._clear_entry_tree()
            self._clear_property_form()
            manager = self._manager()
            index = self.current_trigger_index()
            if manager is None or index is None or index >= len(manager.triggers):
                return
            trigger = manager.triggers[index]

            root = QTreeWidgetItem(["Trigger", self._trigger_label(trigger)])
            root.setData(0, Qt.UserRole, ("trigger", -1))
            self.entry_tree.addTopLevelItem(root)

            for kind, label, entries in (
                ("condition", "Conditions", list(self._read(trigger, "conditions") or [])),
                ("effect", "Effects", list(self._read(trigger, "effects") or [])),
            ):
                group = QTreeWidgetItem([f"{label} ({len(entries)})", ""])
                # Marked rather than identified by row position: reading these
                # back as topLevelItem(1)/(2) works today and breaks the moment
                # the tree grows a row.
                group.setData(0, Qt.UserRole, ("group", kind))
                self.entry_tree.addTopLevelItem(group)
                # GH #213: rows in the game's display order; UserRole stays the list index.
                for entry_index in entry_display_order(trigger, kind) if entries else ():
                    child = QTreeWidgetItem(list(self._describe(kind, entries[entry_index])))
                    child.setData(0, Qt.UserRole, (kind, entry_index))
                    group.addChild(child)
                    self._entry_items[(kind, entry_index)] = child
                group.setExpanded(True)
            self._style_entry_tree(manager, index)
            self._fit_entry_columns()
        finally:
            self._populating = False
        self._set_current_entry_item(self.entry_tree.topLevelItem(0))

    def _vocab_for(self, kind: str):
        """(entries map, presentation map, type attribute) for a kind."""
        if self._vocabulary is None:
            return (None, {}, "condition_type" if kind == "condition" else "effect_type")
        if kind == "condition":
            return (
                self._vocabulary.conditions,
                self._vocabulary.condition_presentation,
                "condition_type",
            )
        return (self._vocabulary.effects, self._vocabulary.effect_presentation, "effect_type")

    def _definition_for(self, kind: str, entry):
        entries, _, type_attribute = self._vocab_for(kind)
        if entries is None:
            return None
        return entries.get(self._read(entry, type_attribute))

    def _describe(self, kind: str, entry) -> tuple[str, str]:
        """(name, detail) for one condition or effect row."""
        _, _, type_attribute = self._vocab_for(kind)
        definition = self._definition_for(kind, entry)
        if definition is None:
            return (f"{kind} type {self._read(entry, type_attribute)}", "")

        details = []
        for attribute in sorted(definition.attributes, key=trigger_fields.form_rank):
            if attribute == type_attribute or attribute in trigger_fields.HIDDEN_FIELDS:
                continue
            value = self._read(entry, attribute)
            # -1 is the library's "unset" sentinel across almost every numeric
            # trigger field, so showing it would bury the fields that are set.
            if value is None or value == -1 or value == "" or value == []:
                continue
            details.append(f"{attribute}={value}")
        return (definition.name.replace("_", " "), ", ".join(details))
