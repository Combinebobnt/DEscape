"""TriggerPanel's structural editing: context menus, select and move requests, trigger and entry ops, and the vocabulary picker."""

from __future__ import annotations

from PyQt5.QtCore import QItemSelection, QItemSelectionModel, Qt, QTimer
from PyQt5.QtWidgets import (
    QMenu,
    QToolButton,
    QTreeWidgetItem,
)

from descape import (
    trigger_organize,
)
from descape.trigger_model import (
    MOVE_FAR_STEPS,
)


class _MoveButton(QToolButton):
    """A ▲▼ button that knows which modifiers its click carried (GH #211:
    Shift moves five places, Ctrl to the top or bottom). Read off the mouse
    release that clicks it, never QApplication.keyboardModifiers(), and
    cleared once the click is delivered, so a keyboard activation or a
    click() reads as a plain one."""

    def __init__(self) -> None:
        super().__init__()
        self.click_modifiers = Qt.NoModifier

    def mouseReleaseEvent(self, event) -> None:
        self.click_modifiers = event.modifiers()
        try:
            # clicked is emitted from inside the base handler.
            super().mouseReleaseEvent(event)
        finally:
            self.click_modifiers = Qt.NoModifier


class _TriggerPanelStructureMixin:
    """TriggerPanel's structural editing and vocabulary picker (trigger_panel.py's docstring has the split's rules)."""

    # -- structural editing (4b.6b) ------------------------------------------
    #
    # Same division as the property form: the panel decides what the user asked
    # for and reports it, the window decides what that means to the model. The
    # panel never adds a trigger itself, so the begin/commit contract stays in
    # one place.

    def refresh_entries(self, select: tuple[str, int] | None = None) -> None:
        """Rebuild the detail tree only, optionally selecting one entry.

        Adding or removing a condition changes one trigger's contents, not the
        trigger list, so rebuilding the whole panel would scroll a 590-trigger
        list back to the top for no reason.

        Recomputes the current trigger's status first (GH #166): an entry
        add, delete, cut, paste or retype never reaches show_scenario().
        """
        manager = self._manager()
        index = self.current_trigger_index()
        if manager is not None and index is not None and index < len(manager.triggers):
            self._recompute_trigger(manager, index)
            self._style_trigger_row(index)
        self._populate_entry_tree()
        if select is not None:
            self.select_entry(*select)

    def select_trigger(self, index: int) -> None:
        """Select the trigger at list `index`, clamped. Used after a
        structural edit to land the user on the trigger that edit produced,
        and after a sort-mode change to keep the same trigger selected across
        rows that just moved.

        `index` is a trigger list index, not a row -- translated through
        `_item_for_index`, populated at the most recent show_scenario(). Falls
        back to the default selection if the index is not currently shown
        (out of range, or -- not reachable today, since nothing filters the
        tree by index -- otherwise absent).
        """
        self.select_triggers([index])

    def _show_trigger_context_menu(self, pos) -> None:
        item = self.tree.itemAt(pos)
        if item is None:
            return
        self.trigger_context_menu(item).exec_(self.tree.viewport().mapToGlobal(pos))

    def _can_edit_now(self) -> bool:
        """_update_buttons()'s can_edit: an editable document, picker not up."""
        picking = self.detail_stack.currentIndex() == self._DETAIL_PICKER
        return self._editable and self._loaded is not None and not picking

    def trigger_context_menu(self, item) -> QMenu:
        """The trigger tree's right-click menu for `item`. Built per click, so
        its enabled states describe that row.

        GH #137's Cut/Copy act on the selection as the right-click left it.
        Paste lands below the clicked row, never the current one; the
        synthetic leading header has no trigger to land below.
        GH #370's Rename opens the GH #279 inline editor on any row, a section header with members too,
        where it edits the title alone."""
        menu = QMenu(self.tree)
        menu.setToolTipsVisible(True)
        can_edit = self._can_edit_now()
        selected = bool(self.selected_trigger_indices())
        index = item.data(self._ROLE_COL, Qt.UserRole)
        cut = menu.addAction("Cut")
        cut.setToolTip("Cut the selected triggers (a collapsed section cuts whole); Paste puts them back with their links")
        cut.setEnabled(can_edit and selected)
        cut.triggered.connect(lambda _=False: self._request_trigger_op("clipboard_cut"))
        copy_action = menu.addAction("Copy")
        copy_action.setToolTip("Copy the selected triggers to the clipboard (a collapsed section copies whole)")
        copy_action.setEnabled(can_edit and selected)
        copy_action.triggered.connect(lambda _=False: self._request_trigger_op("clipboard_copy"))
        paste = menu.addAction("Paste")
        paste.setToolTip(self._clipboard_reason or "Paste the copied triggers below this one, or below a collapsed section's end")
        paste.setEnabled(can_edit and self._clipboard_ready and index is not None)
        paste.triggered.connect(lambda _=False, i=index: self._on_trigger_structural("paste", [self._paste_anchor_index(i)]))
        alone = index is not None and self.selected_trigger_indices() == [index]
        rename = menu.addAction("Rename")
        rename.setToolTip(
            "Rename this trigger in the list (a section header's title, its dashes kept"
            + (")" if alone or index is None else "; select only this trigger)")
        )
        rename.setEnabled(can_edit and alone)
        # Queued, so the editor opens after the menu has closed and handed focus back.
        rename.triggered.connect(lambda _=False, i=index: QTimer.singleShot(0, lambda: self._open_name_editor(i)))
        # GH #211: the ▲▼ verbs and their Ctrl+click, on the selection, each under its own gate.
        of_section = " of Section" if self._grouped else ""
        up, down = self._trigger_move_reasons()
        top, bottom = self._edge_move_reasons()
        _, _, up_tip, down_tip, top_tip, bottom_tip = self._move_tips()
        for text, tip, direction, edge, reason in (
            ("Move Up", up_tip, -1, False, up),
            ("Move Down", down_tip, 1, False, down),
            (f"Move to Top{of_section}", top_tip, -1, True, top),
            (f"Move to Bottom{of_section}", bottom_tip, 1, True, bottom),
        ):
            move = menu.addAction(text)
            move.setToolTip(tip + reason)
            move.setEnabled(can_edit and not reason)
            move.triggered.connect(lambda _=False, d=direction, e=edge: self._request_move(False, d, edge=e))
        # GH #2: on the whole list, so no sort, filter, tag or selection gate.
        renumber_reason = self._renumber_reason()
        renumber = menu.addAction("Renumber IDs to Display Order")
        renumber.setToolTip(self._RENUMBER_TIP + renumber_reason)
        renumber.setEnabled(not renumber_reason)
        renumber.triggered.connect(lambda _=False: self._request_trigger_op("renumber"))
        menu.addSeparator()
        section = menu.addAction("Select Section")
        section.setToolTip("Select this section's divider and every trigger in it")
        section.setEnabled(self._section_for(item) is not None)
        section.triggered.connect(lambda _=False, i=item: self.select_section_of(i))
        tag = menu.addAction("Select Tag")
        tag.setToolTip("Select every trigger carrying this trigger's first tag")
        tag.setEnabled(index is not None and bool(self._row_tags(item)))
        tag.triggered.connect(lambda _=False, i=item: self.select_tag_of(i))
        return menu

    def _show_entry_context_menu(self, pos) -> None:
        item = self.entry_tree.itemAt(pos)
        if item is None:
            return
        self.entry_context_menu(item).exec_(self.entry_tree.viewport().mapToGlobal(pos))

    def entry_context_menu(self, item) -> QMenu:
        """The entry tree's right-click menu for `item`, built per click.

        GH #137's Cut/Copy act on the selected conditions and effects, the
        Duplicate button's gate. Paste goes into this trigger: below the
        clicked row for its own kind, at the end of each list otherwise (a
        heading, the trigger row, or the other kind)."""
        menu = QMenu(self.entry_tree)
        menu.setToolTipsVisible(True)
        trigger_index = self.current_trigger_index()
        can_edit = self._can_edit_now() and trigger_index is not None and len(self.selected_trigger_indices()) <= 1
        entries = can_edit and bool(self.selected_entry_refs())
        data = item.data(0, Qt.UserRole)
        anchor = tuple(data) if data is not None and data[0] in ("condition", "effect") else None
        cut = menu.addAction("Cut")
        cut.setToolTip("Cut the selected conditions and effects, to paste into any trigger")
        cut.setEnabled(entries)
        cut.triggered.connect(lambda _=False: self._request_entry_op("cut"))
        copy_action = menu.addAction("Copy")
        copy_action.setToolTip("Copy the selected conditions and effects, to paste into any trigger")
        copy_action.setEnabled(entries)
        copy_action.triggered.connect(lambda _=False: self._request_entry_op("clipboard_copy"))
        paste = menu.addAction("Paste")
        paste.setToolTip(self._entry_clipboard_reason or "Paste the copied conditions and effects into this trigger")
        paste.setEnabled(can_edit and self._entry_clipboard_ready)
        paste.triggered.connect(lambda _=False, a=anchor, t=trigger_index: self._on_entry_structural(
            "paste", t, "effect", [], -1, anchor=a
        ))
        # GH #213: the ▲▼ buttons' verbs, on the selection, under the same gate.
        # GH #211 adds their Ctrl+click, which that gate covers too.
        for text, tip, op, reason in zip(
            ("Move Up", "Move Down", "Move to Top", "Move to Bottom"),
            (
                self._ENTRY_MOVE_UP_MENU_TIP,
                self._ENTRY_MOVE_DOWN_MENU_TIP,
                self._ENTRY_MOVE_TOP_TIP,
                self._ENTRY_MOVE_BOTTOM_TIP,
            ),
            ("move_up", "move_down", "move_top", "move_bottom"),
            self._entry_move_reasons() * 2,
            strict=True,
        ):
            move = menu.addAction(text)
            move.setToolTip(tip + reason)
            move.setEnabled(not reason)
            move.triggered.connect(lambda _=False, op=op: self._request_entry_op(op))
        menu.addSeparator()
        same = menu.addAction("Select Same Type")
        same.setToolTip("Select every condition or effect of this one's type in this trigger")
        same.setEnabled(self._same_type_refs(item) != [])
        same.triggered.connect(lambda _=False, i=item: self.select_same_type_as(i))
        return menu

    def _same_type_refs(self, item) -> list[tuple[str, int]]:
        """Every visible entry in this trigger of `item`'s kind and type, or
        [] when `item` is not a condition or effect row."""
        data = None if item is None else item.data(0, Qt.UserRole)
        if data is None or data[0] in ("trigger", "group"):
            return []
        kind, entry_index = data
        trigger = self._entries_for([("trigger", -1)])
        if not trigger:
            return []
        _, _, type_attribute = self._vocab_for(kind)
        siblings = list(self._read(trigger[0][2], f"{kind}s") or [])
        if not 0 <= entry_index < len(siblings):
            return []
        wanted = self._read(siblings[entry_index], type_attribute)
        return [
            (kind, index)
            for index, sibling in enumerate(siblings)
            if self._read(sibling, type_attribute) == wanted
            and (row := self._entry_item_for(kind, index)) is not None
            and not row.isHidden()
        ]

    def select_same_type_as(self, item) -> None:
        """Select every entry of the clicked row's type (GH #60's literal ask),
        the clicked row current. Selection only, never model state."""
        refs = self._same_type_refs(item)
        clicked = tuple(item.data(0, Qt.UserRole)) if refs else None
        if clicked in refs:
            refs.remove(clicked)
            refs.insert(0, clicked)
        self.select_entries(refs)

    def _section_for(self, item):
        """The grouped tree's Section `item` belongs to (a header row is its
        own section's), or None when flat."""
        if not self._grouped or item is None:
            return None
        top = item.parent() or item
        header = top.data(self._ROLE_COL, Qt.UserRole)
        if header is None:
            # The synthetic leading header: its section has no divider.
            return next((s for s in self._sections if s.header_index is None), None)
        return next((s for s in self._sections if s.header_index == header), None)

    def select_section_of(self, item) -> None:
        """Select `item`'s whole section: its divider trigger, if it has one,
        and every member. Hidden rows stay out, as for any selection."""
        section = self._section_for(item)
        if section is None:
            return
        indices = ([] if section.header_index is None else [section.header_index]) + list(section.member_indices)
        self.select_triggers([i for i in indices if not self._item_for_index[i].isHidden()])

    def select_tag_of(self, item) -> None:
        """Select every visible trigger carrying `item`'s leading [tag] anywhere
        in its chain, across sections."""
        tags = self._row_tags(item)
        if not tags or item.data(self._ROLE_COL, Qt.UserRole) is None:
            return
        indices = [
            index
            for index in self._display_slots
            if (row := self._item_for_index.get(index)) is not None
            and not row.isHidden()
            and tags[0] in self._row_tags(row)
        ]
        self.select_triggers(indices)

    def _set_current_trigger_item(self, item) -> None:
        """Make `item` current and the whole selection, whatever keys are held.
        Without an explicit command, ExtendedSelection reads the live keyboard
        modifiers, so a Ctrl still down from Ctrl+V would toggle instead."""
        self.tree.setCurrentItem(item, 0, QItemSelectionModel.ClearAndSelect | QItemSelectionModel.Rows)

    def select_triggers(self, indices) -> None:
        """Select every trigger at list `indices`, the first becoming the
        current item. Unresolvable indices are skipped; with none left this
        falls back to the default selection, like select_trigger()."""
        items = [item for item in map(self._item_for_index.get, indices) if item is not None]
        if not items:
            self._select_default_trigger()
            return
        # Current first: setCurrentItem() clears the selection, so the rest
        # are added after it.
        self._set_current_trigger_item(items[0])
        for item in items[1:]:
            item.setSelected(True)

    def select_all(self) -> None:
        """Edit > Select All in Triggers mode (GH #3): every trigger a filter
        leaves visible. Not tree.selectAll(), which skips collapsed sections'
        members. One selection change, not one per row."""
        items = [
            row
            for index in self._display_slots
            if (row := self._item_for_index.get(index)) is not None and not row.isHidden()
        ]
        if not items:
            return
        self._set_current_trigger_item(items[0])
        selection = QItemSelection()
        for item in items[1:]:
            model_index = self.tree.indexFromItem(item)
            selection.select(model_index, model_index)
        self.tree.selectionModel().select(selection, QItemSelectionModel.Select | QItemSelectionModel.Rows)

    def clear_trigger_selection(self) -> None:
        self.tree.clearSelection()

    def move_row(self, trigger_index: int, delta: int) -> None:
        """Patch the tree for a display-order-only move when flat: only the
        moved trigger's row changes, so this is a take/insert of one item
        rather than show_scenario()'s full repopulate -- the difference
        between one row moving and 590 rows rebuilding on
        `old-allies-final-v2` for a single Move Up click.

        Grouped, a move can cross a section boundary -- a parentage change,
        not a same-parent row swap -- so this falls back to a full repopulate
        there. Per the plan's "measure before optimising": no grouped fast
        path is built speculatively, only if repopulate cost is measured and
        shown to matter.

        `delta` is the signed number of display slots: +-1 for Move Up/Down,
        more for GH #211's five-place move. The caller keeps it inside the
        list; a target row outside the tree patches nothing.
        """
        if self._grouped:
            self.show_scenario(self._loaded, self._pending_exec_order)
            self.select_trigger(trigger_index)
            return

        item = self._item_for_index.get(trigger_index)
        if item is None:
            return
        row = self.tree.indexOfTopLevelItem(item)
        target = row + delta
        if not 0 <= target < self.tree.topLevelItemCount():
            return
        self.tree.takeTopLevelItem(row)
        self.tree.insertTopLevelItem(target, item)

        # _item_for_index needs no correction: every item kept its identity,
        # only rows changed, and the dict maps by identity, not row.
        # _display_sections is left alone too: the edge gate reads positions from _display_slots.
        position = self._display_position.get(trigger_index)
        if position is not None:
            landing = position + delta
            if 0 <= landing < len(self._display_slots):
                self._display_slots.insert(landing, self._display_slots.pop(position))
                # Every row in the span shifted one slot, so its position text
                # is stale; nothing else on this fast path would rewrite it.
                with self._batched_trigger_restyle():
                    for slot in range(min(position, landing), max(position, landing) + 1):
                        index = self._display_slots[slot]
                        self._display_position[index] = slot
                        spanned = self._item_for_index.get(index)
                        if spanned is not None:
                            spanned.setText(self._COL_POS, self._position_text(index))

        self._set_current_trigger_item(item)
        self._update_buttons()

    def show_moved_triggers(self, indices, delta: int) -> None:
        """After a move of several slots or to a section edge (GH #211), or
        any File order view ID move (TASK-569, `delta` signed by direction): the
        moved rows selected and the block's leading row (the first going up,
        the last going down) in view, its section expanded. The trigger-tree
        twin of reselect_moved_entries(). Never reveal_trigger(), which
        scrolls the row to the top of the view and so jumps the list on every
        press. Selection only."""
        # File order view lists by ID, which is the list index.
        by_display = self._sort_mode == "display"
        moved = sorted(
            (index for index in indices if index in self._item_for_index),
            key=lambda index: self._display_position.get(index, index) if by_display else index,
        )
        if not moved:
            self.select_triggers(indices)
            return
        lead = self._item_for_index[moved[0] if delta < 0 else moved[-1]]
        position = self._section_position(lead.parent())
        if position is not None:
            lead.parent().setExpanded(True)
            # Explicit: under a filter the toggle handler is suppressed.
            self._collapsed_keys.discard(trigger_organize.section_key(self._sections, position))
        self.select_triggers(indices)
        self.tree.scrollToItem(lead)

    def select_entry(self, kind: str, entry_index: int) -> None:
        self.select_entries([(kind, entry_index)])

    def select_entries(self, refs) -> None:
        """Select exactly these (kind, index) entry rows, the first becoming
        the current one. Selection only: no model state is touched."""
        items = [item for item in (self._entry_item_for(*ref) for ref in refs) if item is not None]
        if not items:
            return
        # Current first: setCurrentItem() clears the selection.
        self._set_current_entry_item(items[0])
        for item in items[1:]:
            item.setSelected(True)

    def reselect_moved_entries(self, refs, delta: int, scroll: int) -> None:
        """After a Move Up/Down rebuilt the entry tree (GH #213): the same
        rows selected, the tree back at vertical scroll value `scroll`, and
        the block's leading row in view, so a repeated click never jumps the
        list. Selection only."""
        self.select_entries(refs)
        self.entry_tree.verticalScrollBar().setValue(scroll)
        moved = self.selected_entry_refs()
        if moved:
            lead = self._entry_item_for(*(moved[0] if delta < 0 else moved[-1]))
            self.entry_tree.scrollToItem(lead)

    def _set_current_entry_item(self, item) -> None:
        """_set_current_trigger_item() for the entry tree: an explicit
        ClearAndSelect, so a held Ctrl cannot turn it into a toggle."""
        if item is None:
            self.entry_tree.setCurrentItem(None)
            return
        self.entry_tree.setCurrentItem(item, 0, QItemSelectionModel.ClearAndSelect | QItemSelectionModel.Rows)

    def _entry_item_for(self, kind: str, entry_index: int):
        """The entry-tree row for one (kind, list index) ref, or None. Never
        by row: the rows sit in display order (GH #213)."""
        return self._entry_items.get((kind, entry_index))

    def current_entry_ref(self) -> tuple[str, int] | None:
        """Public (kind, index) of the current condition or effect, so the
        viewer never reads the live entry object off the panel (GH #59)."""
        return self._selected_entry_ref()

    def picker_showing(self) -> bool:
        """Whether the vocabulary picker page is up in place of the form."""
        return self.detail_stack.currentIndex() == self._DETAIL_PICKER

    def _selected_entry_ref(self) -> tuple[str, int] | None:
        """(kind, index) for the selected condition or effect, or None when the
        selection is the trigger row, a group heading, or nothing."""
        current = self._current_entry()
        if current is None:
            return None
        kind, entry_index, _entry = current
        if kind == "trigger":
            return None
        return (kind, entry_index)

    def set_clipboard_state(self, ready: bool, reason: str = "") -> None:
        """Whether Paste has something to paste (GH #28's button). `reason`
        says why a held clipboard cannot be pasted here (GH #3's version gate);
        the button stays enabled and the window logs it on press."""
        self._clipboard_ready = ready
        self._clipboard_reason = reason
        self.trigger_paste_button.setToolTip(reason or self._PASTE_TOOLTIP)
        self._update_buttons()

    def set_entry_clipboard_state(self, ready: bool, reason: str = "") -> None:
        """set_clipboard_state()'s twin for the entry clipboard (GH #137),
        read by the entry tree's context menu. No button: the entry row is full."""
        self._entry_clipboard_ready = ready
        self._entry_clipboard_reason = reason

    def _move_clicked(self, button: _MoveButton, direction: int, entries: bool) -> None:
        """A ▲▼ click (GH #211): plain is one place, Shift five, Ctrl the top
        or the bottom. Ctrl wins when both are held."""
        modifiers = button.click_modifiers
        self._request_move(
            entries, direction, far=bool(modifiers & Qt.ShiftModifier), edge=bool(modifiers & Qt.ControlModifier)
        )

    def request_move(self, direction: int, far: bool = False, edge: bool = False) -> None:
        """The move shortcuts' entry point (GH #211): up (`direction` < 0) or
        down, one place, five (`far`) or to the top or bottom (`edge`). Moves
        the selected conditions and effects when the entry tree has the
        focus, the selected triggers otherwise, through the buttons' own
        request path and so under the same gate."""
        self._request_move(self.entry_tree.hasFocus(), direction, far, edge)

    def _request_move(self, entries: bool, direction: int, far: bool = False, edge: bool = False) -> None:
        request = self._request_entry_op if entries else self._request_trigger_op
        if edge:
            request("move_top" if direction < 0 else "move_bottom")
        elif far:
            request("move_up" if direction < 0 else "move_down", MOVE_FAR_STEPS)
        else:
            request("move_up" if direction < 0 else "move_down")

    def _request_trigger_op(self, op: str, arg=None) -> None:
        if not self._editable:
            return
        if op in ("move_up", "move_down", "move_top", "move_bottom"):
            # GH #211: gated here, not only by a disabled button, since a shortcut has no button.
            indices = self.selected_trigger_indices()
            if not indices or not self._can_edit_now() or self._move_view_reason():
                return
            step = op in ("move_up", "move_down")
            if step and self._trigger_move_reasons()[0 if op == "move_up" else 1]:
                return
            if self._sort_mode != "display":
                # TASK-569: File order view moves IDs, a signed step count or an end.
                if not step:
                    id_arg = "top" if op == "move_top" else "bottom"
                else:
                    id_arg = (arg or 1) * (-1 if op == "move_up" else 1)
                self._on_trigger_structural("move_ids", indices, id_arg)
                return
            # An edge move that section_edge_refusal() refuses goes on: the window says why in the status bar.
            if arg is None:
                self._on_trigger_structural(op, indices)
            else:
                self._on_trigger_structural(op, indices, arg)
            return
        if op == "new":
            self._on_trigger_structural(op, ())
            return
        if op == "renumber":
            # Re-checked: a menu built before an undo can be stale.
            if not self._renumber_reason():
                self._on_trigger_structural(op, self.selected_trigger_indices())
            return
        if op == "clipboard_copy":
            # Records nothing: the window reads the selection itself.
            self._on_copy_triggers()
            return
        if op == "clipboard_cut":
            self._on_cut_triggers()
            return
        if op == "paste":
            # The anchor, not a selection: the block lands below it.
            current = self.current_paste_anchor()
            self._on_trigger_structural(op, [] if current is None else [current])
            return
        # GH #134: Duplicate and Delete take a collapsed section whole; Move (above) stays header-only.
        indices = self.selected_block_indices() if op in ("copy", "delete") else self.selected_trigger_indices()
        if not indices:
            return
        self._on_trigger_structural(op, indices)

    def _request_entry_op(self, op: str, arg=None) -> None:
        if not self._editable:
            return
        trigger_index = self.current_trigger_index()
        if trigger_index is None:
            return
        if op == "new":
            self._open_picker(trigger_index)
            return
        if op == "clipboard_copy":
            self._on_copy_entries()
            return
        entries = self._entries_for(self.selected_entry_refs())
        if op == "retype":
            # One entry only: the picker offers that entry's own kind.
            if len(entries) != 1:
                return
            kind, entry_index, entry = entries[0]
            _, _, type_attribute = self._vocab_for(kind)
            self._open_picker(
                trigger_index,
                entry_ref=(kind, entry_index),
                entry_type=self._read(entry, type_attribute),
            )
            return
        if not entries:
            return
        # The whole set, as one undo record (GH #60).
        refs = [(kind, entry_index) for kind, entry_index, _entry in entries]
        if op in ("move_up", "move_down", "move_top", "move_bottom"):
            up = op in ("move_up", "move_top")
            # GH #211: gated here, not only by a disabled button, since a shortcut has no button.
            if self._entry_move_reasons()[0 if up else 1]:
                return
            direction = -1 if up else 1
            if op in ("move_top", "move_bottom"):
                self._on_entry_move(trigger_index, refs, direction, True)
            else:
                self._on_entry_move(trigger_index, refs, direction * (arg or 1))
            return
        self._on_entry_structural(op, trigger_index, refs[0][0], refs, -1)

    # -- the vocabulary picker -----------------------------------------------

    def _open_picker(
        self,
        trigger_index: int,
        entry_ref: tuple[str, int] | None = None,
        entry_type: int = -1,
    ) -> None:
        """Show the vocabulary picker over the lower pane.

        `entry_ref` switches it from Add to Change: the picker then offers only
        that entry's own kind and opens on the type it already has.
        """
        if self._vocabulary is None:
            return
        self._picker_trigger_index = trigger_index
        self._picker_entry_ref = entry_ref
        self._picker_entry_type = entry_type if isinstance(entry_type, int) else -1
        self._picker_focus_kind = self._highlighted_entry_kind() if entry_ref is None else None
        self._populate_picker()
        retyping = self._picker_entry_ref is not None
        self.picker_add_button.setText("Change" if retyping else "Add")
        self.picker_tree.setHeaderLabels(
            [f"Change this {self._picker_entry_ref[0]} to"]
            if retyping
            else ["Add to this trigger"]
        )
        self.picker_filter.setPlaceholderText(
            f"Filter {self._picker_entry_ref[0]}s…"
            if retyping
            else "Filter conditions and effects…"
        )
        self.detail_stack.setCurrentIndex(self._DETAIL_PICKER)
        # The picker gets the whole lower pane while it is up. Sharing it with
        # the detail tree left 144 types in five visible rows, and the tree has
        # nothing to offer here anyway: the trigger being added to was latched
        # at open and cannot be changed from it.
        self._set_entry_list_visible(False)
        self.picker_filter.setFocus()
        self._update_buttons()

    def _close_picker(self) -> None:
        """Leave the picker without adding anything.

        Opening and cancelling must be indistinguishable from never having
        opened it: no callback fires from here, so no model is built and no
        record is pushed. commit_trigger_edit() pushes unconditionally, which
        is why the window is only ever told about an accepted pick.
        """
        self._picker_trigger_index = None
        self._picker_entry_ref = None
        self._picker_entry_type = -1
        self._picker_focus_kind = None
        self.picker_tree.clear()
        self.picker_filter.clear()
        self.detail_stack.setCurrentIndex(self._DETAIL_FORM)
        self._set_entry_list_visible(True)
        self._update_buttons()

    def _set_entry_list_visible(self, visible: bool) -> None:
        self.entry_tree.setVisible(visible)
        for i in range(self.entry_buttons.count()):
            widget = self.entry_buttons.itemAt(i).widget()
            if widget is not None:
                widget.setVisible(visible)

    def _populate_picker(self) -> None:
        """Both kinds in Add mode; only the latched entry's own kind in Change
        mode, where it also opens on the type the entry already has. Retyping a
        condition into an effect is not offered: they are separate lists."""
        self.picker_tree.clear()
        if self._vocabulary is None:
            return
        groups = (("condition", "Conditions"), ("effect", "Effects"))
        if self._picker_entry_ref is not None:
            groups = tuple(g for g in groups if g[0] == self._picker_entry_ref[0])
        current_item = None
        for kind, label in groups:
            entries, _presentation, _type_attribute = self._vocab_for(kind)
            if entries is None:
                continue
            # Type 0 is "none", the library's placeholder. The in-game editor
            # offers neither of them -- docs/INGAME_EDITOR_REFERENCE.md's own
            # traversal lists 40 conditions and 100 effects, and "None" is in
            # neither -- and it sits first in an alphabetical list, one click
            # from the top, so offering it is all downside.
            offered = [entry for entry in entries.values() if entry.id != 0]
            group = QTreeWidgetItem([f"{label} ({len(offered)})"])
            # Not Qt.UserRole: _picked() reads None there as "a heading".
            group.setData(0, self._PICKER_KIND_ROLE, kind)
            self.picker_tree.addTopLevelItem(group)
            # Alphabetical rather than by type id: the in-game editor's own New
            # Condition and New Effect lists are alphabetical, and an id order
            # is meaningless to a scenario designer.
            for entry in sorted(offered, key=lambda e: e.name):
                child = QTreeWidgetItem([entry.name.replace("_", " ")])
                child.setData(0, Qt.UserRole, (kind, entry.id))
                group.addChild(child)
                if (
                    self._picker_entry_ref is not None
                    and entry.id == self._picker_entry_type
                ):
                    current_item = child
            group.setExpanded(self._picker_group_open(kind))
        # After the tree is built: setCurrentItem on a child whose group has not
        # been added yet does nothing.
        if current_item is not None:
            self.picker_tree.setCurrentItem(current_item)
            self.picker_tree.scrollToItem(current_item)

    def _highlighted_entry_kind(self) -> str | None:
        """The kind ("condition"/"effect") whose heading or row is the detail
        tree's current item; None for the Trigger row or nothing."""
        item = self.entry_tree.currentItem()
        data = item.data(0, Qt.UserRole) if item is not None else None
        if not data:
            return None
        kind = data[1] if data[0] == "group" else data[0]
        return kind if kind in ("condition", "effect") else None

    def _picker_group_open(self, kind: str) -> bool:
        return self._picker_focus_kind is None or kind == self._picker_focus_kind

    def _apply_picker_filter(self, text: str) -> None:
        """Hide non-matching rows, and any group left with nothing under it.
        A filter opens every group, so a collapsed one cannot swallow a match."""
        needle = text.strip().lower()
        for i in range(self.picker_tree.topLevelItemCount()):
            group = self.picker_tree.topLevelItem(i)
            shown = 0
            for c in range(group.childCount()):
                child = group.child(c)
                hidden = bool(needle) and needle not in child.text(0).lower()
                child.setHidden(hidden)
                shown += not hidden
            group.setHidden(bool(needle) and shown == 0)
            kind = group.data(0, self._PICKER_KIND_ROLE)
            group.setExpanded(bool(needle) or kind is None or self._picker_group_open(kind))

    def _picked(self) -> tuple[str, int] | None:
        item = self.picker_tree.currentItem()
        if item is None or item.isHidden():
            return None
        data = item.data(0, Qt.UserRole)
        if data is None:  # a group heading
            return None
        return data

    def _accept_pick(self) -> None:
        picked = self._picked()
        trigger_index = self._picker_trigger_index
        entry_ref = self._picker_entry_ref
        entry_type = self._picker_entry_type
        if picked is None or trigger_index is None:
            return
        kind, type_id = picked
        self._close_picker()
        if entry_ref is None:
            self._on_entry_structural("new", trigger_index, kind, -1, type_id)
            return
        if type_id == entry_type:
            # The only place a no-op retype can be stopped: commit_trigger_edit()
            # pushes unconditionally, so calling through would record an undo
            # step that changes nothing.
            return
        self._on_entry_structural("retype", trigger_index, kind, entry_ref[1], type_id)
