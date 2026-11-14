"""TriggerPanel's organizing: the tag combo and tag requests, section navigation and
collapse state, and section management (new, rename, move to section)."""

from __future__ import annotations

import dataclasses

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QInputDialog,
    QLineEdit,
    QMessageBox,
)

from descape import (
    trigger_fields,
    trigger_organize,
)
from descape.scenario_io import (
    parse_triggers,
)


class _TriggerPanelOrganizeMixin:
    """TriggerPanel's tags and sections (trigger_panel.py's docstring has the split's rules)."""

    def _populate_tag_combo(self, names: list[str]) -> None:
        """Repopulate the tag facet from every tag in `names`' leading tag chains,
        preserving the current selection across a same-document rebuild (a
        sort-mode toggle, a structural edit's full repopulate) the same way
        the filter text field already does by simply never being cleared."""
        previous = self.tag_combo.currentData() if self.tag_combo.count() else None
        self.tag_combo.blockSignals(True)
        try:
            self.tag_combo.clear()
            self.tag_combo.addItem("All tags", None)
            tags = sorted({tag for name in names for tag in trigger_organize.tag_chain(name)})
            for tag in tags:
                self.tag_combo.addItem(tag, tag)
            self.tag_combo.setEnabled(True)
            restored = self.tag_combo.findData(previous)
            self.tag_combo.setCurrentIndex(restored if restored >= 0 else 0)
        finally:
            self.tag_combo.blockSignals(False)

    def _refresh_tag_combo(self, manager) -> None:
        """Repopulate the facet off live names; refilter only if its tag vanished."""
        previous = self.tag_combo.currentData()
        self._populate_tag_combo([self._read(t, "name") or "" for t in manager.triggers])
        if self.tag_combo.currentData() != previous:
            self._apply_filter()
        self._update_buttons()

    _TAG_ADD_TIP = "Add a tag to every selected trigger that lacks it"
    _TAG_ADD_NONE_TIP = "Select a trigger to tag"
    _TAG_REMOVE_SELECTED_TIP = "Remove a tag from the selected triggers only"
    _TAG_REMOVE_SELECTED_NONE_TIP = "The selected triggers carry no tag"
    _TAG_RENAME_TIP = "Rename the chosen tag on every trigger carrying it, including ones the text filter hides"
    _TAG_REMOVE_ALL_TIP = "Remove the chosen tag from every trigger carrying it, including ones the text filter hides"
    _TAG_PICK_TIP = "Pick a tag in the tag filter first"

    def _tag_count(self, tag: str) -> int:
        """How many triggers carry `tag` anywhere in their tag chain, read off raw names."""
        manager = self._manager()
        if manager is None:
            return 0
        return sum(1 for t in manager.triggers if tag in trigger_organize.tag_chain(self._read(t, "name") or ""))

    @staticmethod
    def _triggers_text(count: int) -> str:
        return f"{count} trigger{'s' if count != 1 else ''}"

    def request_tag_rename(self) -> None:
        """Ask for a new name for the facet's tag, confirm a merge onto an
        existing tag, then report through on_tag_rename."""
        old = self.tag_combo.currentData()
        if old is None or not self._editable:
            return
        count = self._tag_count(old)
        text, ok = QInputDialog.getText(
            self,
            f'Rename tag "{old}"',
            f"{self._triggers_text(count)} carry this tag. New tag:",
            QLineEdit.Normal,
            old,
        )
        new = text.strip()
        if not ok or not new or new == old:
            return
        if self.tag_combo.findData(new) >= 0:
            answer = QMessageBox.question(
                self,
                "Merge tags",
                f'Merge tag "{old}" ({self._triggers_text(count)}) into existing tag '
                f'"{new}" ({self._triggers_text(self._tag_count(new))})? One undo reverses it.',
            )
            if answer != QMessageBox.Yes:
                return
        self._on_tag_rename(old, new)

    def request_tag_remove(self) -> None:
        """Report stripping the facet's tag through on_tag_remove. No confirm
        (GH #101): it is one undo step."""
        tag = self.tag_combo.currentData()
        if tag is None or not self._editable:
            return
        self._on_tag_remove(tag)

    def _selected_tags(self, indices) -> list[str]:
        """The distinct tags `indices` carry, off each row's stashed _TAG_ROLE."""
        items = (self._item_for_index.get(i) for i in indices)
        return sorted({tag for item in items if item is not None for tag in self._row_tags(item)})

    def _row_tags(self, item) -> tuple[str, ...]:
        """`item`'s stashed tag chain, () for an untagged or synthetic row."""
        return tuple(item.data(self._ROLE_COL, self._TAG_ROLE) or ())

    def _stash_tags(self, item, name: str) -> None:
        item.setData(self._ROLE_COL, self._TAG_ROLE, tuple(trigger_organize.tag_chain(name)))

    @staticmethod
    def _selected_text(count: int) -> str:
        return f"{count} selected trigger{'s' if count != 1 else ''}"

    def request_tag_add(self) -> None:
        """Ask for a tag, offering the document's own, and report adding it to
        the selected triggers through on_tag_add (GH #101)."""
        indices = self.selected_trigger_indices()
        if not indices or not self._editable:
            return
        tags = [self.tag_combo.itemData(i) for i in range(1, self.tag_combo.count())]
        current = tags.index(self.current_tag()) if self.current_tag() in tags else 0
        text, ok = QInputDialog.getItem(
            self, "Add tag", f"Add a tag to {self._selected_text(len(indices))}:", tags, current, True
        )
        tag = text.strip()
        if ok and tag:
            self._on_tag_add(indices, tag)

    def request_tag_remove_selected(self) -> None:
        """Report removing one of the selection's tags from the selected
        triggers only, asking which when they carry more than one (GH #101)."""
        indices = self.selected_trigger_indices()
        tags = self._selected_tags(indices)
        if not tags or not self._editable:
            return
        tag = tags[0]
        if len(tags) > 1:
            # Default to the facet's tag, else the first selected row's leading tag, never a sorted-first chained one.
            rows = (self._item_for_index.get(i) for i in indices)
            leading = next(chain[0] for item in rows if item is not None and (chain := self._row_tags(item)))
            preferred = self.current_tag() if self.current_tag() in tags else leading
            text, ok = QInputDialog.getItem(
                self,
                "Remove tag",
                f"Remove a tag from {self._selected_text(len(indices))}:",
                tags,
                tags.index(preferred),
                False,
            )
            if not ok or text not in tags:
                return
            tag = text
        self._on_tag_remove_selected(indices, tag)

    def current_tag(self) -> str | None:
        """The tag facet's selected tag, or None on "All tags"."""
        return self.tag_combo.currentData()

    def set_tag_filter(self, tag: str | None) -> None:
        """Point the tag facet at `tag`; a tag the document lacks is a no-op."""
        found = self.tag_combo.findData(tag)
        if found >= 0:
            self.tag_combo.setCurrentIndex(found)

    # -- section navigation --------------------------------------------------

    # Menu labels are elided at this many average characters: the corpus's
    # p90 divider title is 38 long, its longest 133.
    _SECTION_LABEL_CHARS = 40

    def _populate_sections_menu(self, parts: list[trigger_organize.Section]) -> None:
        """One jump action per section after the fixed entries. A flat tree
        gets none, and Collapse/Expand All are disabled."""
        for action in self._section_actions:
            self.sections_menu.removeAction(action)
            action.deleteLater()
        self._section_actions = []
        metrics = self.sections_menu.fontMetrics()
        width = metrics.averageCharWidth() * self._SECTION_LABEL_CHARS
        for section in parts:
            if section.header_index is None:
                label = self._BEFORE_FIRST_SECTION
                # The synthetic row is not selectable; jump to its first member,
                # the same row _select_default_trigger() lands on.
                target = section.member_indices[0]
            else:
                label = trigger_organize.section_label(section.title)
                target = section.header_index
            text = metrics.elidedText(label, Qt.ElideRight, width).replace("&", "&&")
            action = self.sections_menu.addAction(text)
            action.setData(target)
            action.triggered.connect(lambda checked=False, i=target: self.reveal_trigger(i))
            self._section_actions.append(action)
        self.collapse_all_action.setEnabled(bool(parts))
        self.expand_all_action.setEnabled(bool(parts))

    def _update_section_actions(self) -> None:
        """Disable jumps to a row the filter currently hides."""
        for action in self._section_actions:
            item = self._item_for_index.get(action.data())
            action.setEnabled(item is not None and not item.isHidden())

    def _section_position(self, item) -> int | None:
        """`item`'s index in _sections when it is a grouped top-level row."""
        if not self._grouped or item is None or item.parent() is not None:
            return None
        position = self.tree.indexOfTopLevelItem(item)
        return position if 0 <= position < len(self._sections) else None

    def _on_section_toggled(self, item, collapsed: bool) -> None:
        """Remember a user's collapse or expand of one section. Populate and
        filter passes expand rows as a side effect, so they are ignored."""
        if self._populating or self._filter_expanded:
            return
        position = self._section_position(item)
        if position is None:
            return
        key = trigger_organize.section_key(self._sections, position)
        if collapsed:
            self._collapsed_keys.add(key)
        else:
            self._collapsed_keys.discard(key)

    def _sync_collapsed_keys_from_tree(self) -> None:
        self._collapsed_keys = {
            trigger_organize.section_key(self._sections, position)
            for position in range(min(len(self._sections), self.tree.topLevelItemCount()))
            if not self.tree.topLevelItem(position).isExpanded()
        }

    def _live_sections(self):
        """(triggers, display-order sections) read from the live manager, or
        None with no document."""
        manager = self._manager()
        if manager is None:
            return None
        triggers = manager.triggers
        names = [self._read(t, "name") or "" for t in triggers]
        return triggers, trigger_organize.sections(names, list(manager.trigger_display_order))

    def collapse_snapshot(self) -> list[tuple[object, bool]] | None:
        """Each section's collapse state against its header trigger object
        (None for the leading section), for follow_collapse() once an edit,
        undo or jump has moved the sections. None when nothing is collapsed."""
        if not self._collapsed_keys:
            return None
        live = self._live_sections()
        if live is None:
            return None
        triggers, parts = live
        return [
            (
                None if section.header_index is None else triggers[section.header_index],
                trigger_organize.section_key(parts, position) in self._collapsed_keys,
            )
            for position, section in enumerate(parts)
        ]

    def follow_collapse(self, snapshot: list[tuple[object, bool]] | None) -> None:
        """Re-key the collapse state onto the live sections by header identity
        (TASK-225), so a swap or a duplicate above a same-titled section does
        not hand it another section's state. Call before the rebuild. A header
        the snapshot lacks (a copy, a restored deletion) keeps its title key."""
        if snapshot is None:
            return
        live = self._live_sections()
        if live is None:
            return
        triggers, parts = live
        # The snapshot holds every header object, so these ids cannot be reused meanwhile.
        before = {id(header): collapsed for header, collapsed in snapshot}
        keys = set()
        for position, section in enumerate(parts):
            header = None if section.header_index is None else triggers[section.header_index]
            key = trigger_organize.section_key(parts, position)
            if before.get(id(header), key in self._collapsed_keys):
                keys.add(key)
        self._collapsed_keys = keys

    def collapse_all_sections(self) -> None:
        self._filter_expanded = False
        self.tree.collapseAll()
        self._sync_collapsed_keys_from_tree()
        # Re-expands what an active filter matches, so its hits stay visible.
        self._apply_filter()

    def expand_all_sections(self) -> None:
        self._filter_expanded = False
        self.tree.expandAll()
        self._sync_collapsed_keys_from_tree()
        self._apply_filter()

    def reveal_trigger(self, index: int) -> None:
        """Jump to the trigger at list `index`: expand its section, make it the
        current row and scroll it to the top of the view."""
        item = self._item_for_index.get(index)
        if item is None:
            return
        top = item.parent() or item
        position = self._section_position(top)
        if position is not None:
            top.setExpanded(True)
            # Explicit: under a filter the toggle handler is suppressed.
            self._collapsed_keys.discard(trigger_organize.section_key(self._sections, position))
        self._set_current_trigger_item(item)
        self.tree.scrollToItem(item, QAbstractItemView.PositionAtTop)

    def _retitle_section(self, trigger_index: int, title: str) -> None:
        """Follow a divider renamed to another divider: the menu label and the
        collapse keys, which a label patch alone would leave stale."""
        position = next(
            (p for p, s in enumerate(self._sections) if s.header_index == trigger_index), None
        )
        if position is None or self._sections[position].title == title:
            return
        collapsed = [
            p for p in range(len(self._sections))
            if trigger_organize.section_key(self._sections, p) in self._collapsed_keys
        ]
        self._sections[position] = dataclasses.replace(self._sections[position], title=title)
        self._display_sections = [
            dataclasses.replace(s, title=title) if s.header_index == trigger_index else s
            for s in self._display_sections
        ]
        self._collapsed_keys = {trigger_organize.section_key(self._sections, p) for p in collapsed}
        self._populate_sections_menu(self._sections)

    # -- section management --------------------------------------------------

    _NEW_SECTION_TIP = "Add a section header at the end of the current trigger's section"
    _MOVE_TO_SECTION_TIP = "Move the selected triggers to the end of another section"

    def _section_view_reason(self) -> str:
        """Why the section verbs are off for view reasons, or "": Display
        order (File order view has no sections) and no filter or tag."""
        if self._sort_mode != "display":
            return " (switch to Display order to edit sections)"
        if self.filter_edit.text().strip() or self.tag_combo.currentData() is not None:
            return " (clear the filter or tag to edit sections)"
        return ""

    def _current_name(self) -> str | None:
        index = self.current_trigger_index()
        manager = self._manager()
        if index is None or manager is None or index >= len(manager.triggers):
            return None
        return self._read(manager.triggers[index], "name") or ""

    def _move_targets(self) -> list[tuple[str, int | None, bool]]:
        """(label, header index, enabled) per section the selection can move
        to, in display order. The leading section is offered even when empty,
        which sections() omits. A section already holding every selected
        trigger is disabled."""
        parts = list(self._display_sections)
        if parts and parts[0].header_index is not None:
            parts.insert(0, trigger_organize.Section("", None, ()))
        homes = {self._section_of.get(index) for index in self.selected_trigger_indices()}
        targets = []
        for section in parts:
            header = section.header_index
            label = (
                self._BEFORE_FIRST_SECTION
                if header is None
                else trigger_organize.section_label(section.title)
            )
            targets.append((label, header, homes != {header}))
        return targets

    def _move_blocker(self) -> str:
        """Why Move to Section is off for this selection, or ""."""
        indices = self.selected_trigger_indices()
        if not indices:
            return " (select a trigger first)"
        if any(self._section_of.get(index, -1) == index for index in indices):
            return " (a section header cannot move into another section)"
        if not any(enabled for _, _, enabled in self._move_targets()):
            return " (this file has no other section)"
        return ""

    def _populate_move_menu(self) -> None:
        self.section_move_menu.clear()
        metrics = self.section_move_menu.fontMetrics()
        width = metrics.averageCharWidth() * self._SECTION_LABEL_CHARS
        for label, header, enabled in self._move_targets():
            text = metrics.elidedText(label, Qt.ElideRight, width).replace("&", "&&")
            action = self.section_move_menu.addAction(text)
            # The header index, never the title: two sections can share one.
            action.setData(header)
            action.setEnabled(enabled)
            action.triggered.connect(lambda checked=False, h=header: self.request_move_to_section(h))

    def request_new_section(self) -> None:
        """Ask for a title, then report a new section after the current one."""
        if not self._editable or self._section_view_reason():
            return
        title, ok = QInputDialog.getText(self, "New Section", "Section title:", QLineEdit.Normal, "")
        title = title.strip()
        if not ok or not title:
            return
        current = self.current_trigger_index()
        self._on_trigger_structural("new_section", [] if current is None else [current], title)

    def request_section_rename(self) -> None:
        """Ask for a new title for the current divider and write it through
        the trigger name funnel, keeping the divider's own decoration."""
        index = self.current_trigger_index()
        name = self._current_name()
        if not self._editable or self._section_view_reason() or index is None or name is None:
            return
        parts = trigger_organize.split_divider(name)
        if parts is None:
            return
        text, ok = QInputDialog.getText(
            self, "Rename Section", "Section title:", QLineEdit.Normal, parts[1]
        )
        title = text.strip()
        if not ok or not title or title == parts[1]:
            return
        self.rename_section(index, title)

    def rename_section(self, index: int, title: str) -> None:
        """Retitle the divider at `index`. A title that would not read back
        (one ending in a divider character) is refused before any edit."""
        manager = self._manager()
        if manager is None or not 0 <= index < len(manager.triggers):
            return
        name = self._read(manager.triggers[index], "name") or ""
        if trigger_organize.split_divider(name) is None:
            return
        value = trigger_organize.retitle_divider(name, title)
        reason = trigger_organize.divider_title_error(value, title)
        if reason:
            QMessageBox.warning(self, "Cannot use that section title", f'"{title}": {reason}.')
            return
        self.rename_trigger(index, value)

    def rename_trigger(self, index: int, value: str) -> None:
        """Name the trigger at `index`, `value` verbatim as the Name field
        writes it: one undo step, and nothing recorded if it already reads so."""
        manager = self._manager()
        if manager is None or not 0 <= index < len(manager.triggers):
            return
        name = self._read(manager.triggers[index], "name") or ""
        if value == name:
            return
        # As in _changed(): crossing the divider predicate changes the sections.
        repartitions = trigger_organize.is_divider(name) != trigger_organize.is_divider(value)
        spec = next(s for s in trigger_fields.TRIGGER_FIELDS if s.name == "name")
        self._on_trigger_field(index, spec, value)
        if repartitions:
            self.show_scenario(self._loaded, self._pending_exec_order)
            self.select_trigger(index)
        else:
            self._refresh_labels(index)
        current = self._current_entry()
        if current is not None and current[0] == "trigger":
            self._populate_property_form()

    def _name_editor_text(self, index: int) -> str:
        """What the inline editor edits: a section header's title alone (GH #370), else the whole name."""
        name = self._read(self._manager().triggers[index], "name") or ""
        parts = trigger_organize.split_divider(name)
        return name if parts is None else parts[1]

    def request_trigger_rename(self, item) -> None:
        """GH #279: the double-clicked row's inline editor, for one selected
        trigger on an editable document. A section header with members keeps
        double-click for collapse, and has the menu's Rename (GH #370) and
        Rename Section… for its title."""
        index = item.data(self._ROLE_COL, Qt.UserRole)
        if index is None or item.childCount():
            return
        self._open_name_editor(index)

    def _open_name_editor(self, index: int) -> None:
        """The inline name editor on trigger `index`'s row, for it alone
        selected on an editable document; any row, section headers included,
        which edit their title and keep their dashes."""
        item = self._item_for_index.get(index)
        if item is None or not self._can_edit_now() or self.selected_trigger_indices() != [index]:
            return
        self.tree.scrollToItem(item)
        self.tree.editItem(item, self._COL_NAME)

    def _apply_trigger_rename(self, *_closed) -> None:
        pending, self._name_delegate.pending = self._name_delegate.pending, None
        manager = self._manager()
        if pending is None or manager is None or not 0 <= pending[0] < len(manager.triggers):
            return
        index, text = pending
        if trigger_organize.split_divider(self._read(manager.triggers[index], "name") or "") is None:
            self.rename_trigger(index, text)
        # As the Rename Section dialog reads its title: stripped, and an empty one changes nothing.
        elif title := text.strip():
            self.rename_section(index, title)

    def request_move_to_section(self, header_index: int | None) -> None:
        if not self._editable or self._section_view_reason() or self._move_blocker():
            return
        self._on_trigger_structural("move_to_section", self.selected_trigger_indices(), header_index)

    def _update_section_buttons(self, can_edit: bool) -> None:
        view = self._section_view_reason()
        self.section_new_action.setEnabled(can_edit and not view)
        self.section_new_action.setToolTip(self._NEW_SECTION_TIP + view)
        name = self._current_name() if can_edit and not view else None
        single = len(self.selected_trigger_indices()) == 1
        self.section_rename_action.setEnabled(
            single and name is not None and trigger_organize.split_divider(name) is not None
        )
        blocker = view or self._move_blocker()
        self.section_move_action.setEnabled(can_edit and not blocker)
        self.section_move_action.setToolTip(self._MOVE_TO_SECTION_TIP + blocker)

    def _select_default_trigger(self) -> None:
        """topLevelItem(0), or its first child when that row is the synthetic
        leading header -- selecting the header itself would show an empty
        entry pane on most real files (a before-first run is 1..6 triggers
        and never zero once dividers exist, per the corpus census)."""
        if not self.tree.topLevelItemCount():
            return
        first = self.tree.topLevelItem(0)
        if first.data(self._ROLE_COL, Qt.UserRole) is None and first.childCount():
            first = first.child(0)
        self._set_current_trigger_item(first)

    def _manager(self):
        """The live TriggerManager, re-fetched every call.

        Never cached, for the same process-global poisoning reason
        show_scenario() re-parses. Memoized upstream, so this is a depoison().
        """
        if self._loaded is None:
            return None
        return parse_triggers(self._loaded)
