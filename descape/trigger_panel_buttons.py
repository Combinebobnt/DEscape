"""TriggerPanel's button enablement: which edit and move buttons are live, and the reason each disabled one gives."""

from __future__ import annotations

from descape.trigger_model import (
    id_order_to_edge,
    ids_follow_display_order,
    section_edge_refusal,
    would_move_entries,
)


class _TriggerPanelButtonsMixin:
    """TriggerPanel's button enablement (trigger_panel.py's docstring has the split's rules)."""

    # -- button enablement ---------------------------------------------------

    def _update_buttons(self) -> None:
        """Every button's enabled state, in one place.

        Gated on _editable, which is the panel-populate-time read of
        trigger_write_supported. A read-only file browses exactly as before.
        """
        picking = self.detail_stack.currentIndex() == self._DETAIL_PICKER
        has_document = self._loaded is not None
        selected = len(self.selected_trigger_indices())
        # The trigger verbs act on the selection set. The entry verbs act on
        # the current trigger, so only while it is the one selected.
        has_trigger = self.current_trigger_index() is not None and selected <= 1
        can_edit = self._editable and has_document and not picking

        self.trigger_new_button.setEnabled(can_edit)
        self.trigger_copy_button.setEnabled(can_edit and selected >= 1)
        self.trigger_clipboard_copy_button.setEnabled(can_edit and selected >= 1)
        self.trigger_paste_button.setEnabled(can_edit and self._clipboard_ready)
        self.trigger_delete_button.setEnabled(can_edit and selected >= 1)
        self._update_reorder_buttons(can_edit and selected >= 1)
        self._update_section_move_buttons(can_edit and selected >= 1)
        self._update_section_buttons(can_edit)
        self.sections_button.setEnabled(has_document)

        self.entry_new_button.setEnabled(can_edit and has_trigger and self._vocabulary is not None)
        # Copy and Delete act on the selection set (GH #60); Type… stays single.
        entries = len(self.selected_entry_refs()) if has_trigger else 0
        self.entry_copy_button.setEnabled(can_edit and entries >= 1)
        self.entry_delete_button.setEnabled(can_edit and entries >= 1)
        self.entry_retype_button.setEnabled(
            can_edit and entries == 1 and self._vocabulary is not None
        )
        for button, tip, reason in zip(
            (self.entry_move_up_button, self.entry_move_down_button),
            (self._ENTRY_MOVE_UP_TIP, self._ENTRY_MOVE_DOWN_TIP),
            self._entry_move_reasons(),
            strict=True,
        ):
            button.setEnabled(not reason)
            button.setToolTip(tip + reason)

        self.picker_add_button.setEnabled(picking and self._picked() is not None)

        has_tag = self.tag_combo.currentData() is not None
        self.tags_button.setEnabled(can_edit)
        # A disabled action's tooltip carries the reason, as the Sections actions do.
        self.tag_add_action.setEnabled(can_edit and selected >= 1)
        self.tag_add_action.setToolTip(self._TAG_ADD_TIP if selected >= 1 else self._TAG_ADD_NONE_TIP)
        has_selected_tag = bool(self._selected_tags(self.selected_trigger_indices()))
        self.tag_remove_selected_action.setEnabled(can_edit and has_selected_tag)
        self.tag_remove_selected_action.setToolTip(
            self._TAG_REMOVE_SELECTED_TIP if has_selected_tag else self._TAG_REMOVE_SELECTED_NONE_TIP
        )
        self.tag_rename_action.setEnabled(can_edit and has_tag)
        self.tag_rename_action.setToolTip(self._TAG_RENAME_TIP if has_tag else self._TAG_PICK_TIP)
        self.tag_remove_action.setEnabled(can_edit and has_tag)
        self.tag_remove_action.setToolTip(self._TAG_REMOVE_ALL_TIP if has_tag else self._TAG_PICK_TIP)

        # On has_document, not on can_edit: the variable list is worth reading
        # on a read-only file, and the dialog gates its own Add/Remove on the
        # editable flag it is handed.
        self.variables_button.setEnabled(has_document)
        self.variables_action.setEnabled(self.variables_button.isEnabled())

    # GH #211: the ▲▼ tips name the modifier clicks; a menu entry ignores modifiers, so its tip does not.
    _ENTRY_MOVE_UP_MENU_TIP = "Move the selected conditions and effects up in display order"
    _ENTRY_MOVE_DOWN_MENU_TIP = "Move the selected conditions and effects down in display order"
    _ENTRY_MOVE_UP_TIP = _ENTRY_MOVE_UP_MENU_TIP + ". Shift+click: five places. Ctrl+click: top of the list"
    _ENTRY_MOVE_DOWN_TIP = _ENTRY_MOVE_DOWN_MENU_TIP + ". Shift+click: five places. Ctrl+click: bottom of the list"
    _ENTRY_MOVE_TOP_TIP = "Move the selected conditions and effects to the top of their list"
    _ENTRY_MOVE_BOTTOM_TIP = "Move the selected conditions and effects to the bottom of their list"

    def _entry_move_reasons(self) -> tuple[str, str]:
        """Why Move Entry Up and Move Entry Down are each off, "" when on (GH
        #213). The entry verbs' gate (an editable file, the picker closed, one
        trigger), then an entry row selected that would_move_entries() that way.
        A move to the top or bottom (GH #211) would change something exactly
        when one step that way would, so this is its gate too."""
        if not self._editable or self._loaded is None:
            reason = " (this file's triggers cannot be edited)"
        elif self.detail_stack.currentIndex() == self._DETAIL_PICKER:
            reason = " (close the type picker first)"
        elif self.current_trigger_index() is None or len(self.selected_trigger_indices()) > 1:
            reason = " (select a single trigger)"
        else:
            refs = self.selected_entry_refs()
            # Only with rows selected: this runs on every selection change.
            trigger = self._entries_for([("trigger", -1)]) if refs else []
            if trigger:
                return (
                    "" if would_move_entries(trigger[0][2], refs, -1) else " (already first)",
                    "" if would_move_entries(trigger[0][2], refs, 1) else " (already last)",
                )
            reason = " (select a condition or effect)"
        return (reason, reason)

    # GH #211: the ▲▼ tips name the modifier clicks; a menu entry ignores modifiers, so its tip does not.
    _MOVE_UP_MENU_TIP = "Move the selected trigger up in display order"
    _MOVE_DOWN_MENU_TIP = "Move the selected trigger down in display order"
    _MOVE_UP_TIP = _MOVE_UP_MENU_TIP + ". Shift+click: five places. Ctrl+click: top of its section"
    _MOVE_DOWN_TIP = _MOVE_DOWN_MENU_TIP + ". Shift+click: five places. Ctrl+click: bottom of its section"
    _MOVE_TOP_TIP = "Move the selected triggers to the top of their section, or of the list when there are no sections"
    _MOVE_BOTTOM_TIP = (
        "Move the selected triggers to the bottom of their section, or of the list when there are no sections"
    )

    # TASK-569: the same verbs in File order view move trigger IDs; picked by _move_tips().
    _ID_MOVE_UP_MENU_TIP = (
        "Move the selected triggers to earlier IDs; the triggers they pass move one ID later "
        "and trigger links follow. Display order is unchanged"
    )
    _ID_MOVE_DOWN_MENU_TIP = (
        "Move the selected triggers to later IDs; the triggers they pass move one ID earlier "
        "and trigger links follow. Display order is unchanged"
    )
    _ID_RENUMBER_NOTE = ". Renumber IDs to Display Order, on the right-click menu, does the whole list at once"
    _ID_MOVE_UP_TIP = _ID_MOVE_UP_MENU_TIP + ". Shift+click: five places. Ctrl+click: the first IDs" + _ID_RENUMBER_NOTE
    _ID_MOVE_DOWN_TIP = _ID_MOVE_DOWN_MENU_TIP + ". Shift+click: five places. Ctrl+click: the last IDs" + _ID_RENUMBER_NOTE
    _ID_MOVE_TOP_TIP = "Give the selected triggers the first IDs; trigger links follow and display order is unchanged"
    _ID_MOVE_BOTTOM_TIP = "Give the selected triggers the last IDs; trigger links follow and display order is unchanged"

    def _move_tips(self) -> tuple[str, str, str, str, str, str]:
        """(▲ button, ▼ button, Move Up menu, Move Down menu, Move to Top,
        Move to Bottom) tips for the current sort mode."""
        if self._sort_mode == "display":
            return (
                self._MOVE_UP_TIP,
                self._MOVE_DOWN_TIP,
                self._MOVE_UP_MENU_TIP,
                self._MOVE_DOWN_MENU_TIP,
                self._MOVE_TOP_TIP,
                self._MOVE_BOTTOM_TIP,
            )
        return (
            self._ID_MOVE_UP_TIP,
            self._ID_MOVE_DOWN_TIP,
            self._ID_MOVE_UP_MENU_TIP,
            self._ID_MOVE_DOWN_MENU_TIP,
            self._ID_MOVE_TOP_TIP,
            self._ID_MOVE_BOTTOM_TIP,
        )

    def _move_view_reason(self) -> str:
        """Why no trigger can move for view reasons, or "". In Display order
        view a move permutes display order; in File order view (TASK-569) it
        moves trigger IDs and keeps display order. Either way the view must be
        unfiltered: a text or 4c tag filter is hide-only, so the adjacent row
        may be hidden and a swap into it would look like nothing happened.
        """
        if self.filter_edit.text().strip() or self.tag_combo.currentData() is not None:
            return " (clear the filter or tag to reorder)"
        return ""

    _RENUMBER_TIP = (
        "Give every trigger a new ID in display order, so File order matches the list. "
        "A file that runs triggers by ID (Legacy) then runs them in the listed order. Trigger links follow."
    )

    def _renumber_reason(self) -> str:
        """Why Renumber IDs to Display Order is off, "" when on (GH #2): the
        edit gate, then a display order that is not already identity. Reads
        `_display_slots`, which a one-row move keeps current."""
        if not self._editable or self._loaded is None:
            return " (this file's triggers cannot be edited)"
        if self.detail_stack.currentIndex() == self._DETAIL_PICKER:
            return " (close the type picker first)"
        if ids_follow_display_order(self._display_slots):
            return " (IDs already follow display order)"
        return ""

    def _trigger_move_reasons(self) -> tuple[str, str]:
        """Why Move Up and Move Down are each off, "" when on: the view gate,
        then the block bounds. The leading member must have a slot above it,
        the trailing one a slot below.

        Bounds come from `_display_position`/`_display_slots` (adjacency in
        display order), not tree geometry: grouped, the adjacent row in
        `self.tree` may belong to a different section's top-level item, where
        `indexOfTopLevelItem` would misread it as -1 or as an unrelated row.
        In File order view (TASK-569) the bounds are list indices, which are
        the IDs.
        """
        view = self._move_view_reason()
        if view:
            return view, view
        if self._sort_mode != "display":
            count = self._trigger_count
            positions = [index for index in self.selected_trigger_indices() if index < count]
            if not positions:
                return " (select a trigger first)", " (select a trigger first)"
            return (
                "" if min(positions) > 0 else " (already first)",
                "" if max(positions) < count - 1 else " (already last)",
            )
        positions = [
            self._display_position[index]
            for index in self.selected_trigger_indices()
            if index in self._display_position
        ]
        if not positions:
            return " (select a trigger first)", " (select a trigger first)"
        return (
            "" if min(positions) > 0 else " (already first)",
            "" if max(positions) < len(self._display_slots) - 1 else " (already last)",
        )

    def _edge_move_reasons(self) -> tuple[str, str]:
        """Why Move to Top and Move to Bottom are each off, "" when on (GH
        #211): the view gate, then trigger_model.section_edge_refusal(), whose
        wording the window's status line repeats. Positions come from
        `_display_slots`, which a one-row move keeps current; the cached
        partition supplies membership only."""
        view = self._move_view_reason()
        if view:
            return view, view
        indices = self.selected_trigger_indices()
        if not indices:
            return " (select a trigger first)", " (select a trigger first)"
        if self._sort_mode != "display":
            # File order view moves IDs and has no sections (TASK-569).
            count = self._trigger_count
            if not all(index < count for index in indices):
                return " (select a trigger first)", " (select a trigger first)"
            identity = list(range(count))
            return (
                "" if id_order_to_edge(count, indices, True) != identity else " (already first)",
                "" if id_order_to_edge(count, indices, False) != identity else " (already last)",
            )
        top = section_edge_refusal(self._display_slots, self._display_sections, indices, True)
        bottom = section_edge_refusal(self._display_slots, self._display_sections, indices, False)
        return (f" ({top})" if top else "", f" ({bottom})" if bottom else "")

    def _update_reorder_buttons(self, base_ok: bool) -> None:
        """Move Up/Down follow _trigger_move_reasons(): both disable with a
        tooltip saying why, rather than silently clearing the sort/filter for
        the user. A button stays on while its Ctrl+click edge move is allowed
        (a split block against the top still gathers); _request_trigger_op()
        refuses the step itself."""
        up, down = self._trigger_move_reasons()
        # Only when a step is refused: each edge check builds a whole display order.
        edge_up, edge_down = self._edge_move_reasons() if (up or down) else ("", "")
        up_on = base_ok and not (up and edge_up)
        down_on = base_ok and not (down and edge_down)
        self.trigger_move_up_button.setEnabled(up_on)
        self.trigger_move_down_button.setEnabled(down_on)
        # TASK-365: on only for the gather, so the note says what a click there still does.
        if up and up_on:
            up = up[:-1] + ": Ctrl+click gathers the selection here)"
        if down and down_on:
            down = down[:-1] + ": Ctrl+click gathers the selection here)"
        up_tip, down_tip = self._move_tips()[:2]
        self.trigger_move_up_button.setToolTip(up_tip + up)
        self.trigger_move_down_button.setToolTip(down_tip + down)

    _SECTION_UP_TIP = "Move the selected trigger's whole section above the section before it"
    _SECTION_DOWN_TIP = "Move the selected trigger's whole section below the section after it"

    def _section_move_reasons(self) -> tuple[str, str]:
        """Why Section Up and Section Down are each off, "" when on (GH
        #133). Display order and no filter or tag, plus a grouped tree, a
        selection inside one real section, and a real neighbouring section
        that way. The leading run never moves and is never swapped past.
        Bounds come from the display-order partition, never tree geometry."""
        if self._sort_mode != "display":
            reason = " (switch to Display order to reorder)"
            return reason, reason
        if self.filter_edit.text().strip() or self.tag_combo.currentData() is not None:
            reason = " (clear the filter or tag to reorder)"
            return reason, reason
        if not self._grouped:
            return " (this file has no sections)", " (this file has no sections)"
        homes = {self._section_of.get(index) for index in self.selected_trigger_indices()}
        if len(homes) != 1:
            return " (select triggers in one section)", " (select triggers in one section)"
        (home,) = homes
        if home is None:
            reason = " (the triggers before the first section are not a section)"
            return reason, reason
        parts = self._display_sections
        position = next(p for p, s in enumerate(parts) if s.header_index == home)
        up = "" if position > 0 and parts[position - 1].header_index is not None else " (no section above this one)"
        down = "" if position + 1 < len(parts) else " (no section below this one)"
        return up, down

    def _update_section_move_buttons(self, base_ok: bool) -> None:
        """Section Up/Down: off with a tooltip saying why, like Move Up/Down."""
        up, down = self._section_move_reasons()
        self.section_up_button.setEnabled(base_ok and not up)
        self.section_down_button.setEnabled(base_ok and not down)
        self.section_up_button.setToolTip(self._SECTION_UP_TIP + up)
        self.section_down_button.setToolTip(self._SECTION_DOWN_TIP + down)
