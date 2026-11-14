"""TriggerPanel's property form: a field widget per entry field, the instruction and
button-grid previews, and reporting an edit."""

from __future__ import annotations

import math
from typing import ClassVar

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QLabel,
    QLineEdit,
    QWidget,
)

from descape import (
    button_grid,
    civ_art,
    messages_fields,
    object_catalog,
    train_locations,
    trigger_fields,
    trigger_geometry,
    trigger_organize,
    unit_references,
)
from descape.button_grid import ButtonGrid
from descape.facet_preview import FacetPreview
from descape.instruction_preview import InstructionPreview
from descape.text_edits import ProseTextEdit, XsTextEdit, _MultiLineEdit
from descape.value_picker import ValueLineEdit
from descape.viewer_common import (
    _fit_combo_width,
    _IndeterminateMixin,
    _relabel_player_rows,
)

# trigger_fields._ENUM_TYPES' key for players.PlayerId; PlayerColorId is a colour, not a player.
PLAYER_ID_PRESENTATION = "PlayerId"


def _same_shown(a, b) -> bool:
    """Whether a form row shows the same value, NaN matching NaN."""
    return a == b or (isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b))


class _TriggerPanelFormMixin:
    """TriggerPanel's property form and edit reporting (trigger_panel.py's docstring has the split's rules)."""

    # -- the property form ---------------------------------------------------

    def _clear_property_form(self) -> None:
        """Empty the form, unparenting each widget immediately.

        setParent(None) rather than deleteLater() alone: deleteLater defers
        destruction to the next event-loop pass, so the old rows stay visible
        and keep their geometry, and the next populate draws its rows on top of
        them. That renders as every label overlapping every other one, which no
        widget-level assertion notices.

        takeRow() rather than takeAt(): takeAt empties a row but keeps it, so
        every populate stacked another set of empty rows. Not removeRow(),
        which deletes synchronously, and a row's own signal can be what
        triggered this repopulate.
        """
        self._rows = []
        self._form_refs = None
        self._unit_ref_rows = {}
        self._pick_buttons = {}
        self._type_pick_rows = {}
        self._group_rows = {}
        if self._instruction_preview is not None:
            self._instruction_preview.detach()
            # A rebuild sends the new view in this same event, so nothing paints between them.
            self.on_instruction_view(None)
        self._instruction_preview = None
        if self._facet_preview is not None:
            self._facet_preview.detach()
        self._facet_preview = None
        self._facet_preview_field = ""
        if self._button_grid is not None:
            self._button_grid.detach()
        self._button_grid = None
        while self.property_form.rowCount():
            row = self.property_form.takeRow(0)
            # A spanning addRow(widget) row fills only one of the two.
            for item in (row.labelItem, row.fieldItem):
                if item is not None:
                    self._discard_layout_item(item)

    @classmethod
    def _discard_layout_item(cls, item) -> None:
        """Unparent and defer-delete a taken item's widget, or every widget in
        its layout (a row whose field is a layout)."""
        widget = item.widget()
        if widget is not None:
            # Hidden first: a row added this same gesture still has Qt's queued
            # show pending, which would map it as a top-level window that takes focus.
            widget.hide()
            widget.setParent(None)
            widget.deleteLater()
            return
        layout = item.layout()
        if layout is not None:
            while layout.count():
                cls._discard_layout_item(layout.takeAt(0))
            layout.deleteLater()

    def refresh_portraits(self) -> None:
        """GH #363: redraws the open form's button grid and instruction preview after the portrait files change,
        or a player's portrait civ (GH #372)."""
        if self._button_grid is not None:
            self._button_grid.refresh()
        if self._instruction_preview is not None:
            self._instruction_preview.refresh()

    def refresh_player_labels(self, labels, colors) -> None:
        """GH #130: store the labels for the next form build and relabel the
        PlayerId combos already on screen, keeping their "(differs)" and "unknown" rows."""
        self._player_labels = labels
        self._player_colors = colors
        if self._instruction_preview is not None:
            self._instruction_preview.set_player_colors(colors)
        for spec, _kind, _index, widget in self._rows:
            if spec.presentation == PLAYER_ID_PRESENTATION and isinstance(widget, QComboBox):
                _relabel_player_rows(widget, labels, colors)

    def _current_entry(self) -> tuple[str, int, object] | None:
        """(kind, entry index, live object) for whatever the detail tree has
        selected, or None."""
        item = self.entry_tree.currentItem()
        if item is None:
            return None
        data = item.data(0, Qt.UserRole)
        if data is None:
            return None
        kind, entry_index = data
        if kind == "group":
            # A Conditions/Effects heading. It carries its kind for the button
            # logic, but it is not an entry and has no fields of its own.
            return None
        manager = self._manager()
        trigger_index = self.current_trigger_index()
        if manager is None or trigger_index is None or trigger_index >= len(manager.triggers):
            return None
        trigger = manager.triggers[trigger_index]
        if kind == "trigger":
            return (kind, -1, trigger)
        entries = list(self._read(trigger, f"{kind}s") or [])
        if not 0 <= entry_index < len(entries):
            return None
        return (kind, entry_index, entries[entry_index])

    def selected_entry_refs(self) -> list[tuple[str, int]]:
        """(kind, index) for every selected condition or effect, in tree
        order: conditions then effects, each in display order (GH #213).

        Drops the trigger row, the group headings and hidden rows (_picked()'s
        guard), so a row the user cannot see is never swept into a bulk verb.
        """
        rows = []
        for item in self.entry_tree.selectedItems():
            data = item.data(0, Qt.UserRole)
            if data is None or item.isHidden() or data[0] in ("trigger", "group"):
                continue
            rows.append((self._entry_row_path(item), data))
        return [ref for _path, ref in sorted(rows)]

    def _form_refs_now(self) -> tuple[tuple[str, int], ...]:
        """The refs the property form should show: the selected entries, or
        the current row alone when none is selected (the trigger row's form)."""
        refs = self.selected_entry_refs()
        if refs:
            return tuple(refs)
        current = self._current_entry()
        return () if current is None else ((current[0], current[1]),)

    def _entries_for(self, refs) -> list[tuple[str, int, object]]:
        """(kind, index, live object) per ref, dropping any that no longer
        resolves against the current trigger. ("trigger", -1) is the trigger."""
        manager = self._manager()
        trigger_index = self.current_trigger_index()
        if manager is None or trigger_index is None or trigger_index >= len(manager.triggers):
            return []
        trigger = manager.triggers[trigger_index]
        resolved = []
        for kind, entry_index in refs:
            if kind == "trigger":
                resolved.append((kind, -1, trigger))
                continue
            entries = list(self._read(trigger, f"{kind}s") or [])
            if 0 <= entry_index < len(entries):
                resolved.append((kind, entry_index, entries[entry_index]))
        return resolved

    def _on_entry_selected(self) -> None:
        """currentItemChanged."""
        self._sync_entry_form()

    def _on_entry_selection_changed(self) -> None:
        """itemSelectionChanged: a Ctrl+click can change the set without
        moving the current row, so the form follows the set."""
        self._sync_entry_form()

    def _sync_entry_form(self) -> None:
        """Rebuild the form for the current selection, once per gesture.

        A click emits both entry-tree signals, and so does
        _populate_entry_tree()'s end-of-populate setCurrentItem() (#27 Trap 5).
        The latch lives here, not in _populate_property_form(), which the
        refresh paths call directly and must never skip.
        """
        if self._populating:
            return
        if self._form_refs_now() != self._form_refs:
            self._populate_property_form()
        self._update_buttons()
        self.on_focus_changed()

    def _specs_for(self, kind: str, entry) -> tuple:
        if kind == "trigger":
            if self._loaded is None:
                return trigger_fields.TRIGGER_FIELDS
            # By version, as the vocabulary is picked: a field the file's format lacks gets no row.
            return trigger_fields.trigger_specs(self._loaded.scenario_version)
        definition = self._definition_for(kind, entry)
        if definition is None:
            return ()
        _, presentation, type_attribute = self._vocab_for(kind)
        specs = trigger_fields.field_specs(definition, presentation, type_attribute)
        if kind == "effect":
            # Which cluster slot is authoritative is a runtime decision, and
            # writing an inert one can corrupt the live one.
            specs = trigger_fields.apply_quantity_cluster_rule(specs, entry)
        # One field order across every form (GH #41, GH #278).
        return trigger_fields.form_order(specs)

    def _populate_property_form(self) -> None:
        self._populating = True
        refs = self._form_refs_now()
        selection = (self.current_trigger_index(), refs)
        if selection != self._list_types_refs:
            # GH #41: a remembered object list type lasts only while the same entries stay selected.
            self._list_types = {}
            self._list_types_refs = selection
        try:
            self._clear_property_form()
            entries = self._entries_for(refs)
            self._form_refs = refs
            if not entries:
                return
            # N = 1 is the same code with a one-entry intersection (GH #60).
            specs = trigger_fields.shared_specs([self._specs_for(k, entry) for k, _i, entry in entries])
            if not specs:
                empty = QLabel(self._no_fields_text(len(entries)))
                # Wrapped: the N-entry wording is wider than the pane's minimum.
                empty.setWordWrap(True)
                self.property_form.addRow(empty)
                return
            objects = [entry for _k, _i, entry in entries]
            kind, entry_index, _entry = entries[0]
            if all(k == "effect" and self._read(e, "effect_type") == self._CREATE_OBJECT_EFFECT for k, _i, e in entries):
                # GH #59: the stamping tool works but users don't find it.
                hint = QLabel(self._CREATE_OBJECTS_HINT)
                hint.setToolTip(self._CREATE_OBJECTS_HINT_TIP)
                hint.setWordWrap(True)
                self.property_form.addRow(hint)
            names = {s.name for s in specs}
            area_grid = all(f in names for f in trigger_geometry.AREA_FIELDS)
            area_cells = []
            for spec in specs:
                # Compared as shown, so a sentinel BOOL's -1 and 0 share one unticked box.
                value = trigger_fields.shared_value(
                    objects, spec.attribute, lambda e, a, s=spec: trigger_fields.form_value(s, self._read(e, a))
                )
                widget = self._build_widget(spec, refs, value)
                if trigger_fields.has_list_type(spec) and isinstance(widget, ValueLineEdit):
                    self._add_list_type_row(spec, widget, value)
                    self._add_type_pick_button(widget, spec, refs)
                if area_grid and spec.name in trigger_geometry.AREA_FIELDS:
                    # GH #277: the game's boxes. The tooltip keeps the field's name, X1 alone doesn't say area.
                    label = QLabel(spec.name.rpartition("_")[2].upper())
                    label.setBuddy(widget)
                    label.setToolTip(spec.label)
                    widget.setToolTip(spec.label)
                    area_cells.append((label, widget))
                    if spec.name == trigger_geometry.AREA_FIELDS[-1]:
                        self.property_form.addRow(self._area_grid(area_cells))
                elif spec.kind == trigger_fields.STR and spec.multiline:
                    # GH #38/#139: caption on its own row, the box (or a locked field's text) full width below.
                    label = QLabel(spec.label)
                    label.setBuddy(widget)
                    self.property_form.addRow(label)
                    self.property_form.addRow(widget)
                else:
                    # One-line fields keep the side caption (GH #139 decision, 2026-10-05).
                    self.property_form.addRow(spec.label, widget)
                    label = self.property_form.labelForField(widget)
                if spec.tooltip:
                    widget.setToolTip(spec.tooltip)
                    if label is not None:
                        label.setToolTip(spec.tooltip)
                self._rows.append((spec, kind, entry_index, widget))
                if kind == "effect" and spec.name in self._FACET_UNITS and self._facet_preview is None:
                    # GH #167: its own row directly under the facet field.
                    self._facet_preview = FacetPreview()
                    self._facet_preview_field = spec.name
                    self.property_form.addRow(self._facet_preview)
                button_grid_effect = self._button_grid_effect(entries) if spec.name == "button_location" else ""
                if button_grid_effect and self._button_grid is None:
                    # GH #169: the building's panel, directly under the slot it sets.
                    self._button_grid = ButtonGrid()
                    self.property_form.addRow(self._button_grid)
                if len(refs) == 1 and kind != "trigger":
                    self._add_group_row(spec, specs, refs)
            self._update_group_rows()
            if self._facet_preview is not None:
                fields = {spec.name: widget for spec, _kind, _index, widget in self._rows}
                facet = self._facet_preview_field
                single = _entry if len(entries) == 1 else None
                self._facet_preview.bind(
                    fields, facet, self._FACET_UNITS[facet], self._read, single, self._fit_property_height
                )
            if self._button_grid is not None:
                fields = {spec.name: widget for spec, _kind, _index, widget in self._rows}
                self._button_grid.bind(
                    fields,
                    self._button_grid_effect(entries),
                    self._read,
                    _entry,
                    self._fit_property_height,
                    self._button_grid_context(entries[0][1]),
                )
            if len(entries) == 1 and kind == "effect" and self._read(_entry, "effect_type") == self._DISPLAY_INSTRUCTIONS:
                self._add_instruction_preview(_entry)
        finally:
            self._populating = False
        self._fit_property_height()

    # The viewer's own _CREATE_OBJECT_EFFECT, which stamp_create_objects() checks.
    _CREATE_OBJECT_EFFECT = 11
    _DISPLAY_INSTRUCTIONS = 20
    # GH #167: Create Object's facet and Replace Object's facet2, each with the unit it turns.
    _FACET_UNITS: ClassVar[dict[str, str]] = {"facet": "object_list_unit_id", "facet2": "object_list_unit_id_2"}

    def _button_grid_effect(self, entries) -> str:
        """The button grid's effect name for a one-effect form of a type it
        covers (GH #169), or "" for anything else."""
        if len(entries) != 1:
            return ""
        kind, _index, entry = entries[0]
        if kind != "effect":
            return ""
        definition = self._definition_for(kind, entry)
        name = "" if definition is None else definition.name
        return name if name in self._BUTTON_GRID_TYPES else ""

    _BUTTON_GRID_TYPES = frozenset(button_grid.EFFECT_MOVERS)

    def _button_grid_context(self, effect_index: int):
        """GH #230: source_player -> the grid's PlayerContext, leaving out
        the effect being edited (current trigger, `effect_index`)."""
        manager = self._manager()
        trigger_index = self.current_trigger_index()
        if manager is None or trigger_index is None or trigger_index >= len(manager.triggers):
            return None
        exclude = (manager.triggers[trigger_index].trigger_id, effect_index)
        return lambda player: train_locations.player_context(self._manager(), self._loaded, player, exclude)

    def _add_instruction_preview(self, entry) -> None:
        """GH #140: a full-width, observe-only preview row under the fields."""
        preview = InstructionPreview()
        self.property_form.addRow(preview)
        manager = self._manager()
        names = [] if manager is None else [label for label, _id in object_catalog.variable_choices(manager)]
        fields = {spec.name: widget for spec, _kind, _index, widget in self._rows}
        preview.bind(
            fields, entry, self._read, names, self._player_colors, self._fit_property_height, self._emit_instruction_view,
            portrait_civ_for=lambda player: civ_art.player_portrait_civ(self._loaded, player),
        )
        self._instruction_preview = preview

    def _emit_instruction_view(self, view) -> None:
        self.on_instruction_view(view)

    # One line at 340 px: a wrapped row is squeezed when the form is populated before it shows.
    _CREATE_OBJECTS_HINT = "Tip: stamp it with Create Objects."
    _CREATE_OBJECTS_HINT_TIP = (
        "Create Objects is on the toolbar. With this effect selected, each click on the map adds "
        "one copy of it per tile under the brush, as one undo step."
    )

    @staticmethod
    def _no_fields_text(count: int) -> str:
        """Why the form is empty. A mixed selection is not refused; it says
        what it found (GH #60)."""
        if count == 1:
            return "This type has no editable fields."
        return f"No fields are shared by the {count} selected entries."

    def _fit_property_height(self) -> None:
        """Tell the scroll area how tall the wrapped form actually is.

        WrapLongRows makes a row's height depend on the width it is given, and
        a QScrollArea sizes its widget from sizeHint(), which is computed as if
        nothing wrapped. The area therefore hands the host the unwrapped height
        and every wrapped row draws over the next one -- rows visibly on top of
        each other, while host width, scrollbar state and field widths all
        still measure correct. Asking the layout for heightForWidth at the real
        viewport width is what makes the vertical scrollbar appear instead.
        """
        # activate() first, or this measures the *previous* form: the rows were
        # added moments ago and Qt has not laid them out yet, so heightForWidth
        # answers for whatever the form held before. That is how this first
        # landed as a 72 px minimum on a form needing 454.
        # New rows wait on Qt's queued show and count as empty until then (102 px
        # on a form needing 694); show now what that queued call would show.
        for child in self.property_host.findChildren(QWidget, options=Qt.FindDirectChildrenOnly):
            if child.isHidden() and not child.testAttribute(Qt.WA_WState_ExplicitShowHide):
                child.show()
        self.property_form.activate()
        height = self.property_form.minimumSize().height()
        width = self.property_area.viewport().width()
        if width > 0 and self.property_form.hasHeightForWidth():
            height = max(height, self.property_form.heightForWidth(width))
        self.property_host.setMinimumHeight(height)

    def _display_text(self, spec, value) -> str:
        """A read-only field's text, in the same terms its editor would use."""
        if value is trigger_fields.MIXED:
            return self._DIFFERS
        if value is None or value == trigger_fields.UNSET:
            return "(unset)"
        if spec.kind == trigger_fields.INT_LIST:
            return trigger_fields.format_int_list(value) or "(unset)"
        if spec.multiline == trigger_fields.XS:
            return trigger_fields.xs_to_display(value)
        if spec.multiline == trigger_fields.PROSE:
            return messages_fields.normalize_for_display("" if value is None else str(value))[0]
        if spec.kind == trigger_fields.BOOL:
            return "yes" if trigger_fields.checkbox_checked(spec, value) else "no"
        if isinstance(value, float):
            return f"{value:g}"
        if spec.kind == trigger_fields.ENUM:
            for label, choice in spec.choices:
                if choice == value:
                    return f"{label} ({choice})"
        if spec.kind == trigger_fields.REFERENCE:
            if spec.presentation == "Unit":
                resolved = self.describe_unit_reference(value)
                if resolved.startswith(f"{value}:"):  # a dangling id already leads with itself
                    return resolved
                return f"{value}: {resolved}" if resolved else str(value)
            resolved = trigger_fields.resolve_reference(spec.presentation, value)
            return f"{resolved} ({value})" if resolved else str(value)
        return str(value)

    def _build_widget(self, spec, refs, value) -> QWidget:
        """One form row's editor, for the whole selection (GH #60).

        `value` is what every selected entry holds, or MIXED when they differ.
        An indeterminate widget shows "(differs)" and writes nothing until the
        user changes it, gated per widget kind. The gate is the whole risk:
        editingFinished fires on a bare focus-out, so an ungated blank box
        would write "" or [] to every selected entry for merely tabbing past.
        """
        indeterminate = value is trigger_fields.MIXED
        multi = len(refs) > 1
        editable = self._editable and not spec.read_only

        # Every locked field renders the same way, whatever its kind. A
        # disabled combo instead would show its first choice as though it were
        # the value: an inert armour_attack_class read "WONDER (0)" while the
        # int beside it, locked by the very same rule, honestly said "(unset)".
        if spec.read_only:
            label = QLabel(self._display_text(spec, value))
            label.setEnabled(False)
            return label

        if spec.kind == trigger_fields.BOOL:
            widget = QCheckBox()
            if indeterminate:
                # stateChanged, not toggled: PartiallyChecked already counts as
                # checked, so the first click would emit no toggled at all.
                widget.setTristate(True)
                widget.setCheckState(Qt.PartiallyChecked)
            else:
                widget.setChecked(trigger_fields.checkbox_checked(spec, value))
            widget.setEnabled(editable)
            widget.stateChanged.connect(
                lambda state, s=spec, r=refs, w=widget: self._checkbox_changed(s, r, w, state)
            )
            return widget

        if spec.kind == trigger_fields.STR and spec.multiline == trigger_fields.XS:
            widget = XsTextEdit("" if indeterminate else trigger_fields.xs_to_display(value))
            widget.setReadOnly(not editable)
            if indeterminate:
                widget.setPlaceholderText(self._DIFFERS)
            widget.editingFinished.connect(lambda s=spec, r=refs, w=widget: self._xs_changed(s, r, w))
            self._wire_text_box(widget, spec)
            return widget

        if spec.kind == trigger_fields.STR and spec.multiline == trigger_fields.PROSE:
            display, token = messages_fields.normalize_for_display(
                "" if indeterminate or value is None else str(value)
            )
            widget = ProseTextEdit(display, token)
            widget.setReadOnly(not editable)
            if indeterminate:
                widget.setPlaceholderText(self._DIFFERS)
            widget.editingFinished.connect(lambda s=spec, r=refs, w=widget: self._prose_changed(s, r, w))
            self._wire_text_box(widget, spec)
            return widget

        if spec.kind == trigger_fields.STR:
            widget = QLineEdit("" if indeterminate or value is None else str(value))
            widget.setReadOnly(not editable)
            if indeterminate:
                widget.setPlaceholderText(self._DIFFERS)
            # Show the start of the value, not its end. A freshly-set QLineEdit
            # leaves the cursor past the last character, so a name too long for
            # the field rendered as "ure: armour split".
            widget.setCursorPosition(0)
            widget.editingFinished.connect(
                lambda s=spec, r=refs, w=widget, m=multi: self._text_changed(s, r, w, m)
            )
            return widget

        if spec.kind == trigger_fields.INT_LIST:
            widget = QLineEdit("" if indeterminate else trigger_fields.format_int_list(value))
            widget.setReadOnly(not editable)
            widget.setPlaceholderText(self._DIFFERS if indeterminate else "comma separated")
            widget.editingFinished.connect(
                lambda s=spec, r=refs, w=widget, m=multi: self._list_changed(s, r, w, m)
            )
            if spec.presentation != unit_references.LIST_PRESENTATION or multi:
                return widget
            return self._unit_reference_host(spec, refs, widget, editable)

        if spec.kind == trigger_fields.ENUM:
            if trigger_fields.wants_picker(spec):
                return self._build_enum_picker_widget(spec, refs, value, editable)
            widget = QComboBox()
            _fit_combo_width(widget)
            self._add_differs_item(widget, indeterminate)
            for label, choice in spec.choices:
                widget.addItem(f"{label} ({choice})", choice)
            if spec.presentation == PLAYER_ID_PRESENTATION:
                _relabel_player_rows(widget, self._player_labels, self._player_colors)
            if not indeterminate and widget.findData(value) < 0 and isinstance(value, int):
                # A value outside the shipped enum. Kept as its own row rather
                # than snapped to the nearest legal one, which would rewrite a
                # field the user never touched.
                widget.addItem(f"unknown ({value})", value)
            widget.setCurrentIndex(0 if indeterminate else max(widget.findData(value), 0))
            widget.setEnabled(editable)
            widget.currentIndexChanged.connect(lambda _, s=spec, r=refs, w=widget: self._combo_changed(s, r, w))
            return widget

        if spec.kind == trigger_fields.REFERENCE:
            catalog_presentation = trigger_fields.CATALOG_PRESENTATIONS.get(spec.presentation)
            if catalog_presentation is not None:
                return self._build_catalog_widget(spec, refs, value, catalog_presentation, editable)
            if spec.presentation in trigger_fields.DOCUMENT_REFERENCES:
                return self._build_document_reference_widget(spec, refs, value, editable)
            # Unit: a placed-unit reference_id, resolved against the open
            # document (describe_unit_reference) and pickable from the map.
            spin = self._trigger_spinbox(value, editable)
            self._connect_spinbox(spin, spec, refs)
            return self._unit_reference_host(spec, refs, spin, editable)

        if spec.kind == trigger_fields.FLOAT:
            spin = self._trigger_float_spinbox(value, editable)
            self._connect_spinbox(spin, spec, refs)
            return spin

        spin = self._trigger_spinbox(value, editable, unset=spec.sentinel is not None)
        self._connect_spinbox(spin, spec, refs)
        return spin

    def _wire_text_box(self, widget: _MultiLineEdit, spec) -> None:
        """GH #139: apply the stored height for this field and report a drag.
        Keyed by field name, so every effect's `message` shares one height."""
        key = f"trigger.{spec.name}"
        stored = self.text_box_lines(key)
        if stored is not None:
            widget.set_visible_lines(stored)
        # A re-fit per step, or the taller box draws over the rows below it.
        widget.linesChanged.connect(lambda _lines: self._fit_property_height())
        widget.linesCommitted.connect(lambda lines, k=key: self.on_text_box_lines(k, lines))
        widget.linesReset.connect(lambda k=key: self.on_text_box_lines(k, None))

    def _connect_spinbox(self, spin, spec, refs) -> None:
        """Report a spinbox's edits. A blank (indeterminate) one leaves that
        state on its first real change, and also commits from editingFinished
        when the user typed the parked value, which emits no valueChanged."""
        blank = isinstance(spin, _IndeterminateMixin) and spin.is_indeterminate()
        if not blank:
            spin.valueChanged.connect(lambda new, s=spec, r=refs: self._changed(s, r, new))
            return
        spin.valueChanged.connect(lambda new, s=spec, r=refs, w=spin: self._spinbox_changed(s, r, w, new))
        spin.editingFinished.connect(lambda s=spec, r=refs, w=spin: self._spinbox_typed(s, r, w))

    # -- reporting an edit ---------------------------------------------------

    def _changed(self, spec, refs, value) -> None:
        """The one place a widget signal becomes a reported edit, for one
        entry or a selection of them (GH #60).

        The equality check is what actually fires in practice: editingFinished
        arrives on every focus-out whether the text changed or not, and
        commit_trigger_edit() pushes unconditionally, so without it tabbing
        through the form would record one phantom undo step per field. Plural,
        it is "at least one selected entry differs": a value the first entry
        already holds must still reach the others.

        A widget built for a different selection than the live one writes
        nothing: its captured refs are stale, so it cannot say what it edits.

        `_populating` is defence in depth and is unreachable today: every widget
        in _build_widget() is connected *after* its value is set, so a populate
        emits nothing. It is kept because that ordering is the only thing making
        it unreachable, and a future edit that connects before setting would
        otherwise report one edit per field with no other guard in the way. A
        spinbox cannot hold None and a combo cannot hold an out-of-enum value,
        so those populates would differ from the live value and sail past the
        equality check.
        """
        if self._populating or not self._editable or spec.read_only:
            return
        if value is trigger_fields.MIXED:
            return
        if len(self.selected_trigger_indices()) > 1:
            return
        live = self._form_refs_now()
        if tuple(refs) != live:
            return
        trigger_index = self.current_trigger_index()
        entries = self._entries_for(live)
        if trigger_index is None or not entries:
            return
        kind = entries[0][0]
        targets = [(k, i) for k, i, entry in entries if self._read(entry, spec.attribute) != value]
        if not targets:
            return
        before = self._read(entries[0][2], spec.attribute)
        slots_before = self._quantity_slots(entries)
        rows_before = self._form_snapshot(entries)

        # 4c: a rename that crosses the divider predicate changes the section
        # partition, which a label patch can't reflect -- escalate to a full
        # repopulate instead. Checked before the edit is applied, since after
        # it `entry`'s own name has already become the new one.
        repartitions = (
            kind == "trigger"
            and spec.name == "name"
            and trigger_organize.is_divider(before or "") != trigger_organize.is_divider(value or "")
        )

        if kind == "trigger":
            self._on_trigger_field(trigger_index, spec, value)
        else:
            # Only the entries that differ, so Effect.quantity's setter is not
            # re-run on an entry already holding the value.
            self._on_entry_field(trigger_index, targets, spec, value)

        if repartitions:
            self.show_scenario(self._loaded, self._pending_exec_order)
            self.select_trigger(trigger_index)
        else:
            self._refresh_labels(trigger_index)

        entries_after = self._entries_for(live)
        rows_after = self._form_snapshot(entries_after)
        # TASK-561: a row the library derives from another field (item_id) moved with the edit.
        expected = dict(rows_before)
        if spec.name in expected:
            expected[spec.name] = trigger_fields.form_value(spec, value)
        rows_moved = rows_after.keys() != expected.keys() or any(
            not _same_shown(rows_after[name], shown) for name, shown in expected.items()
        )
        if rows_moved or self._quantity_slots(entries_after) != slots_before:
            # Deferred: this runs inside the editing widget's own signal, and
            # repopulating unparents that widget immediately.
            QTimer.singleShot(0, lambda r=live: self.repopulate_form_if_current(r))

    @staticmethod
    def _quantity_slots(entries) -> list[str]:
        """Each effect's live quantity-cluster slot, to spot an edit that
        switched one (the form's locked fields then change)."""
        return [trigger_fields.live_quantity_slot(entry) if kind == "effect" else "" for kind, _i, entry in entries]

    def _form_snapshot(self, entries) -> dict[str, object]:
        """What each row shows, read as _populate_property_form() reads it. The
        quantity cluster is left out: one library value sits behind its rows, and
        _quantity_slots() owns them."""
        objects = [entry for _k, _i, entry in entries]
        return {
            spec.name: trigger_fields.shared_value(
                objects, spec.attribute, lambda e, a, s=spec: trigger_fields.form_value(s, self._read(e, a))
            )
            for spec, *_ in self._rows
            if spec.name not in trigger_fields.CLUSTER_FIELDS
        }

    def repopulate_form_if_current(self, refs) -> None:
        """The cluster switch's or a derived row's deferred form rebuild, and the
        window's after a group write (GH #138), skipped if the user has selected
        something else in between. Keyboard focus in the form stays on its row."""
        if self._form_refs_now() != tuple(refs):
            return
        focus = QApplication.focusWidget()
        focused = None
        if focus is not None:
            focused = next(
                (spec.name for spec, *_, widget in self._rows if widget is focus or widget.isAncestorOf(focus)), None
            )
        scroll = self.property_area.verticalScrollBar().value()
        self._populate_property_form()
        self.property_area.verticalScrollBar().setValue(scroll)
        widget = next((w for spec, *_, w in self._rows if spec.name == focused), None) if focused else None
        if widget is None:
            return
        target = widget.line_edit if isinstance(widget, ValueLineEdit) else widget
        if target.focusPolicy() == Qt.NoFocus:
            # A composite row: its first editor, not the container.
            target = next((w for w in target.findChildren(QWidget) if w.focusPolicy() != Qt.NoFocus), target)
        target.setFocus(Qt.OtherFocusReason)
        self.property_area.ensureWidgetVisible(widget)

    def _list_changed(self, spec, refs, widget: QLineEdit, multi: bool = False) -> None:
        if self._populating:
            return
        if multi and not widget.isModified():
            # The STR rule: a blank "(differs)" box must not write [] on a focus-out.
            return
        try:
            value = trigger_fields.parse_int_list(widget.text())
        except ValueError:
            # Rejected whole rather than half-written: "4, x" must not become
            # [4] and silently drop the user's second value.
            objects = [entry for _k, _i, entry in self._entries_for(self._form_refs_now())]
            shared = trigger_fields.shared_value(objects, spec.attribute, self._read)
            widget.setText("" if shared is trigger_fields.MIXED else trigger_fields.format_int_list(shared))
            return
        self._changed(spec, refs, value)

    def _xs_changed(self, spec, refs, widget: XsTextEdit) -> None:
        # Untouched text is a guaranteed no-op, whatever separators the stored
        # value holds; _changed()'s equality check alone misses CRLF and LF.
        # A "(differs)" box latches "", so leaving it blank writes nothing.
        text = widget.toPlainText()
        if text == widget.latched_text:
            return
        self._changed(spec, refs, trigger_fields.xs_from_display(text))
        widget.latched_text = text

    def _prose_changed(self, spec, refs, widget: ProseTextEdit) -> None:
        # Same latch as _xs_changed(): on a CR- or CRLF-stored field the
        # display text differs from the stored value, so _changed()'s equality
        # check alone would record a phantom undo step on a bare focus-out.
        text = widget.toPlainText()
        if text == widget.latched_text:
            return
        value = messages_fields.substitute_newlines(text, widget.newline_token)
        self._changed(spec, refs, value)
        widget.latched_text = text

    def _refresh_labels(self, trigger_index: int) -> None:
        """Update only the rows an edit can have changed.

        Rebuilding wholesale on every keystroke-commit would scroll a
        590-trigger list back to the top and drop the user's selection.
        """
        self._populating = True
        try:
            manager = self._manager()
            if manager is None or trigger_index >= len(manager.triggers):
                return
            trigger = manager.triggers[trigger_index]
            item = self._item_for_index.get(trigger_index)
            if item is not None:
                item.setText(self._COL_NAME, self._trigger_label(trigger))
                # The tag facet's stashed role, kept in step with a rename that
                # doesn't cross the divider predicate (repartitions handles
                # that case with a full repopulate instead -- see _changed()).
                old_tags = self._row_tags(item)
                self._stash_tags(item, self._read(trigger, "name") or "")
                if self._row_tags(item) != old_tags:
                    # GH #101: a rename can add or retire a tag, so the facet's list follows now.
                    self._refresh_tag_combo(manager)
                if self._grouped and item.parent() is None:
                    self._retitle_section(trigger_index, (self._read(trigger, "name") or "").strip())

            # Every selected row plus the current one: a group edit (GH #60)
            # changes N rows' detail text at once.
            items = list(self.entry_tree.selectedItems())
            current = self.entry_tree.currentItem()
            if current is not None and current not in items:
                items.append(current)
            for entry_item in items:
                data = entry_item.data(0, Qt.UserRole)
                if data is None or data[0] == "group":
                    continue
                kind, entry_index = data
                if kind == "trigger":
                    entry_item.setText(1, self._trigger_label(trigger))
                    continue
                entries = list(self._read(trigger, f"{kind}s") or [])
                if 0 <= entry_index < len(entries):
                    name, detail = self._describe(kind, entries[entry_index])
                    entry_item.setText(0, name)
                    entry_item.setText(1, detail)

            # GH #166: the whole trigger, since one field can flip its rollup, root and group rows.
            self._recompute_trigger(manager, trigger_index)
            self._style_trigger_row(trigger_index)
            if self.current_trigger_index() == trigger_index:
                self._style_entry_tree(manager, trigger_index)
        finally:
            self._populating = False

    def refresh_trigger(self, trigger_index: int) -> None:
        """The window's after a write the panel did not make itself (GH #138's
        group Set and Reset): that trigger's labels and status rows."""
        self._refresh_labels(trigger_index)
