"""TriggerPanel's reference fields: Unit and Unit[] references, map and type picks,
the GH #138 coordinate groups and the catalog, enum and list-type pickers."""

from __future__ import annotations

import dataclasses

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from descape import (
    object_catalog,
    trigger_fields,
    trigger_geometry,
    ui_icons,
    unit_references,
)
from descape.constant_picker import CatalogLineEdit
from descape.value_picker import PickerItem, ValueLineEdit
from descape.viewer_common import (
    TOOL_EYEDROPPER,
    IndeterminateDoubleSpinBox,
    _fit_combo_width,
    _IndeterminateMixin,
    _make_spinbox,
)

# GH #138's coordinate groups, keyed by the field whose row the group's own row follows.
GROUP_LOCATION, GROUP_AREA, GROUP_OBJECTS = "location", "area", "objects"
_GROUP_AFTER_FIELD = {
    trigger_geometry.LOCATION_FIELDS[-1]: (GROUP_LOCATION, trigger_geometry.LOCATION_FIELDS),
    trigger_geometry.AREA_FIELDS[-1]: (GROUP_AREA, trigger_geometry.AREA_FIELDS),
    trigger_geometry.SELECTED_OBJECTS_FIELD: (GROUP_OBJECTS, (trigger_geometry.SELECTED_OBJECTS_FIELD,)),
}
# The map pick a tile group's Set arms: a target tuple's sixth slot.
PICK_POINT, PICK_RECT = "point", "rect"
# An object field's eyedropper, same slot: a unit pick that copies the clicked unit's type (GH #276).
PICK_TYPE = "type"
_TYPE_PICK_TIP = "Click a unit on the map to copy its type here; Esc to cancel"
_GROUP_TEXT = {
    # caption, Set tooltip, Go to tooltip, Reset tooltip
    GROUP_LOCATION: (
        "Location",
        "Click a tile on the map to set the location; Esc to cancel",
        "Centre the map on the location",
        "Clear the location",
    ),
    GROUP_AREA: (
        "Area",
        "Drag a rectangle on the map to set the area; Esc to cancel",
        "Centre the map on the area",
        "Clear the area",
    ),
    GROUP_OBJECTS: (
        "Objects",
        "Click units on the map to add or remove them; Esc to finish",
        "Centre the map between the selected objects",
        "Clear the selected objects",
    ),
}


@dataclasses.dataclass
class _GroupRow:
    """One group's Set / Go to / Reset buttons and what they act on."""

    fields: tuple[str, ...]
    refs: tuple
    set_button: QToolButton
    go_button: QToolButton
    reset_button: QToolButton


class _TriggerPanelFieldsMixin:
    """TriggerPanel's reference fields and map picks (trigger_panel.py's docstring has the split's rules)."""

    # -- Unit / Unit[] references -------------------------------------------

    # A Unit[] label shows at most this many ids, then "... and N more" (the corpus max is 8).
    UNIT_LIST_LINES = 8

    def _unit_list_text(self, ids) -> str:
        lines = [self.describe_unit_reference(ref_id) for ref_id in ids[: self.UNIT_LIST_LINES]]
        if len(ids) > self.UNIT_LIST_LINES:
            lines.append(f"... and {len(ids) - self.UNIT_LIST_LINES} more")
        return "\n".join(lines)

    def _unit_reference_host(self, spec, refs, editor, editable: bool) -> QWidget:
        """The editor and a Pick button on one line, what the id(s) name below.
        The label follows the spinbox live, a list line edit on commit."""
        host = QWidget()
        column = QVBoxLayout(host)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(2)
        row = QHBoxLayout()
        row.addWidget(editor, stretch=1)
        column.addLayout(row)
        label = QLabel()
        label.setWordWrap(True)
        column.addWidget(label)
        is_list = isinstance(editor, QLineEdit)

        def refresh(label=label, editor=editor) -> None:
            if is_list:
                try:
                    ids = trigger_fields.parse_int_list(editor.text())
                except ValueError:
                    return
                label.setText(self._unit_list_text(ids))
            elif isinstance(editor, _IndeterminateMixin) and editor.is_indeterminate():
                label.setText("")
            else:
                label.setText(self.describe_unit_reference(editor.value()))
            label.setVisible(bool(label.text()))

        refresh()
        if is_list:
            # After _list_changed's own connection, so a rejected text is already put back.
            editor.editingFinished.connect(refresh)
        else:
            editor.valueChanged.connect(lambda _new: refresh())
        self._unit_ref_rows[spec.name] = (spec, refs, editor, refresh)
        if spec.name != trigger_geometry.SELECTED_OBJECTS_FIELD:
            # Selected objects' Pick is the Objects group row's Set (GH #138).
            self._add_pick_button(row, spec, refs, editable)
        return host

    def _pick_entry(self, refs, editable: bool) -> tuple[int, str, int] | None:
        """(trigger, entry kind, entry index) for a map pick, which names one editable entry only."""
        trigger_index = self.current_trigger_index()
        if not editable or len(refs) != 1 or refs[0][0] == "trigger" or trigger_index is None:
            return None
        return (trigger_index, *refs[0])

    def _add_pick_button(self, row, spec, refs, editable: bool) -> None:
        """Pick from map, for one editable entry's field only: the window
        arms a map picker whose clicks come back through apply_picked_reference()."""
        entry = self._pick_entry(refs, editable)
        if entry is None:
            return
        target = (*entry, spec.name, spec.presentation == unit_references.LIST_PRESENTATION)
        button = QToolButton()
        button.setText("Pick")
        button.setCheckable(True)
        button.setChecked(target == self._armed_pick)
        button.setToolTip(
            "Click units on the map to add or remove them; Esc to finish"
            if target[4]
            else "Click a unit on the map to set this field; Esc to cancel"
        )
        button.toggled.connect(lambda checked, t=target: self.on_pick_unit(t if checked else None))
        self._pick_buttons[target] = button
        row.addWidget(button)

    def _add_type_pick_button(self, widget: ValueLineEdit, spec, refs) -> None:
        """GH #276: the eyedropper after an object field's "…". It arms the unit
        picker, and the click comes back through apply_picked_type()."""
        entry = self._pick_entry(refs, self._editable)
        if entry is None:
            return
        target = (*entry, spec.name, False, PICK_TYPE)
        button = QToolButton()
        button.setIcon(ui_icons.tool_icon(TOOL_EYEDROPPER))
        button.setCheckable(True)
        button.setToolTip(_TYPE_PICK_TIP)
        button.setAccessibleName("Copy type from map")
        button.setFixedHeight(widget.browse_button.sizeHint().height())
        # Checked before connecting: a populate must not re-fire the arm request.
        button.setChecked(target == self._armed_pick)
        button.toggled.connect(lambda checked, t=target: self.on_pick_unit(t if checked else None))
        self._pick_buttons[target] = button
        self._type_pick_rows[spec.name] = widget
        widget.layout().addWidget(button)

    def refresh_pick_icons(self) -> None:
        """Re-ink the form's eyedroppers after a theme change: built per populate, apply_icons() can't name them."""
        for target, button in self._pick_buttons.items():
            if target[5:] == (PICK_TYPE,):
                button.setIcon(ui_icons.tool_icon(TOOL_EYEDROPPER))

    @staticmethod
    def _area_grid(cells) -> QWidget:
        """GH #277: (caption, box) cells two to a row, each caption above its
        box. Added in field order, which is also the Tab order."""
        host = QWidget()
        grid = QGridLayout(host)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        for position, (label, widget) in enumerate(cells):
            row, column = divmod(position, 2)
            # A spinbox's minimum hint is sized for 2^31, which overflows the pane at larger fonts.
            widget.setMinimumWidth(1)
            grid.addWidget(label, row * 2, column)
            grid.addWidget(widget, row * 2 + 1, column)
        return host

    def _add_group_row(self, spec, specs, refs) -> None:
        """GH #138: the Set / Go to / Reset row after a group's last field, for
        one condition or effect. Set and Reset need an editable file; Go to does not."""
        found = _GROUP_AFTER_FIELD.get(spec.name)
        trigger_index = self.current_trigger_index()
        if found is None or trigger_index is None:
            return
        group, fields = found
        names = {s.name for s in specs}
        if not all(f in names for f in fields):
            return
        if group == GROUP_OBJECTS and spec.presentation != unit_references.LIST_PRESENTATION:
            return
        kind, entry_index = refs[0]
        if group == GROUP_OBJECTS:
            # The unit picker's own target, so its Esc, list toggle and disarm sites carry over.
            target = (trigger_index, kind, entry_index, spec.name, True)
        else:
            target = (trigger_index, kind, entry_index, group, False, PICK_POINT if group == GROUP_LOCATION else PICK_RECT)
        caption, set_tip, go_tip, reset_tip = _GROUP_TEXT[group]
        host = QWidget()
        row = QHBoxLayout(host)
        row.setContentsMargins(0, 0, 0, 0)
        buttons = []
        for text, tip in (("Set", set_tip), ("Go to", go_tip), ("Reset", reset_tip)):
            button = QToolButton()
            button.setText(text)
            button.setToolTip(tip)
            row.addWidget(button)
            buttons.append(button)
        row.addStretch(1)
        set_button, go_button, reset_button = buttons
        set_button.setCheckable(True)
        set_button.setEnabled(self._editable)
        # Checked before connecting: a populate must not re-fire the arm request.
        set_button.setChecked(target == self._armed_pick)
        set_button.toggled.connect(lambda checked, t=target: self.on_pick_unit(t if checked else None))
        self._pick_buttons[target] = set_button
        go_button.clicked.connect(lambda _c=False, t=(trigger_index, kind, entry_index, group): self.on_go_to_group(t))
        reset_button.clicked.connect(lambda _c=False, g=group: self._reset_group(g))
        self.property_form.addRow(caption, host)
        self._group_rows[group] = _GroupRow(fields, tuple(refs), set_button, go_button, reset_button)
        # Typing in the group's own editors never repopulates, so their edits re-gate Go to and Reset.
        for row_spec, _kind, _index, widget in self._rows:
            if row_spec.name not in fields:
                continue
            if group == GROUP_OBJECTS:
                editor = self._unit_ref_rows[row_spec.name][2]
                editor.editingFinished.connect(self._update_group_rows)
            elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                widget.valueChanged.connect(lambda _v: self._update_group_rows())

    def _group_state(self, group: str, row: _GroupRow) -> tuple[bool, bool] | None:
        """(Go to possible, anything to reset) for the entry's live values."""
        entries = self._entries_for(row.refs)
        if not entries:
            return None
        entry = entries[0][2]
        if group == GROUP_OBJECTS:
            value = self._read(entry, row.fields[0])
            ids = [i for i in (value or ()) if isinstance(i, int) and i != trigger_fields.UNSET]
            return any(self.unit_reference_tile(i) is not None for i in ids), bool(value)
        coords = []
        for name in row.fields:
            value = self._read(entry, name)
            coords.append(value if isinstance(value, int) else trigger_fields.UNSET)
        # trigger_geometry's rule: a half-set group draws nothing, so there is nowhere to go.
        return all(c >= 0 for c in coords), any(c != trigger_fields.UNSET for c in coords)

    def _update_group_rows(self) -> None:
        for group, row in self._group_rows.items():
            state = self._group_state(group, row)
            if state is None:
                continue
            can_go, has_value = state
            row.go_button.setEnabled(can_go)
            row.reset_button.setEnabled(self._editable and has_value)

    def _reset_group(self, group: str) -> None:
        """Reset: every field of the group unset (objects: an empty list), one record."""
        row = self._group_rows.get(group)
        trigger_index = self.current_trigger_index()
        if row is None or not self._editable or trigger_index is None or self._form_refs_now() != row.refs:
            return
        if group == GROUP_OBJECTS:
            values = {row.fields[0]: []}
        else:
            values = dict.fromkeys(row.fields, trigger_fields.UNSET)
        self._on_entry_field_group(trigger_index, row.refs[0], values, f"Reset {_GROUP_TEXT[group][0]}", True)

    def set_pick_armed(self, target: tuple | None) -> None:
        """The window's arm state, shown on the Pick buttons without re-firing on_pick_unit."""
        self._armed_pick = target
        for button_target, button in self._pick_buttons.items():
            button.blockSignals(True)
            button.setChecked(button_target == target)
            button.blockSignals(False)

    def unit_reference_ids(self, target: tuple) -> tuple[int, ...]:
        """The ids `target`'s field holds now, or () when the form shows something else."""
        row = self._unit_ref_rows.get(target[3])
        if row is None or not self._pick_target_is_current(target):
            return ()
        spec, refs, _editor, _refresh = row
        entries = self._entries_for(refs)
        if not entries:
            return ()
        value = self._read(entries[0][2], spec.attribute)
        values = value if isinstance(value, (list, tuple)) else (value,)
        return tuple(v for v in values if isinstance(v, int) and v != trigger_fields.UNSET)

    def is_pick_target_current(self, target: tuple) -> bool:
        """Whether the form still shows `target`'s entry alone, for the window's tile picks."""
        return self._pick_target_is_current(target)

    def _pick_target_is_current(self, target: tuple) -> bool:
        # Indexed: a tile pick's target carries a sixth slot, its pick kind (GH #138).
        trigger_index, kind, entry_index = target[:3]
        return (
            self.current_trigger_index() == trigger_index
            and len(self.selected_trigger_indices()) <= 1
            and self._form_refs_now() == ((kind, entry_index),)
        )

    def apply_picked_reference(self, target: tuple, ref_id: int) -> bool:
        """Write a map pick through the form's own funnel, so it is one undo
        record like a typed edit. A scalar field takes the id; a list toggles
        it. Returns whether the picker stays armed: only a list pick on the
        still-current target does. A stale target writes nothing."""
        row = self._unit_ref_rows.get(target[3])
        if row is None or not self._editable or not self._pick_target_is_current(target):
            return False
        spec, refs, editor, refresh = row
        if not target[4]:
            editor.setValue(ref_id)
            return False
        current = list(self.unit_reference_ids(target))
        new = [i for i in current if i != ref_id] if ref_id in current else [*current, ref_id]
        editor.setText(trigger_fields.format_int_list(new))
        self._list_changed(spec, refs, editor)
        refresh()
        self._update_group_rows()
        return True

    def apply_picked_type(self, target: tuple, const: int) -> bool:
        """GH #276: a clicked unit's type into an object row, through commit_value()
        so the row's listeners hear it as they hear Browse. Returns whether the
        target was still live; a stale one writes nothing."""
        row = self._type_pick_rows.get(target[3])
        if row is None or not self._editable or not self._pick_target_is_current(target):
            return False
        row.commit_value(const)
        return True

    def type_pick_caption(self, target: tuple) -> str:
        """The row's own caption, for the armed hint: "source player unit", not the field name."""
        return next((spec.label for spec, *_ in self._rows if spec.name == target[3]), target[3].replace("_", " "))

    def refresh_unit_reference_labels(self, recompute_status: bool = True) -> None:
        """Re-describe every Unit/Unit[] label, after units moved or changed hands.

        Also recomputes the status of every trigger whose entries hold a
        placed-unit id (GH #166), unless `recompute_status` is False: the
        window passes False outside Triggers mode, so a Units-mode edit costs
        nothing here and the mode entry's show_scenario() recomputes instead.
        """
        for _spec, _refs, _editor, refresh in self._unit_ref_rows.values():
            refresh()
        self._update_group_rows()
        if not self._status:
            return
        if not recompute_status:
            self._status_stale = True
            return
        manager = self._manager()
        if manager is None:
            return
        context = self._status_context(manager)
        edited = self.edited_trigger_indices()
        current = self.current_trigger_index()
        with self._batched_trigger_restyle():
            for index in sorted(self._unit_ref_triggers):
                if index < len(manager.triggers):
                    clean = edited is not None and index not in edited
                    self._recompute_trigger(manager, index, context, clean=clean)
                    self._style_trigger_row(index)
        if current in self._unit_ref_triggers:
            self._style_entry_tree(manager, current)

    def _spinbox_changed(self, spec, refs, spin, value) -> None:
        spin.clear_indeterminate()
        self._changed(spec, refs, value)

    def _spinbox_typed(self, spec, refs, spin) -> None:
        if not spin.typed_parked_value():
            return
        spin.clear_indeterminate()
        self._changed(spec, refs, spin.value())

    def _checkbox_changed(self, spec, refs, widget, state) -> None:
        """A tristate box writes once it leaves PartiallyChecked, and never
        goes back: "the selection disagrees" is not a value to pick."""
        if state == Qt.PartiallyChecked:
            return
        widget.setTristate(False)
        self._changed(spec, refs, int(state == Qt.Checked))

    def _combo_changed(self, spec, refs, widget) -> None:
        """The "(differs)" row carries None, which no field holds, so leaving
        it selected (or going back to it) writes nothing."""
        data = widget.currentData()
        if data is not None:
            self._changed(spec, refs, data)

    def _text_changed(self, spec, refs, widget, multi: bool) -> None:
        """A STR commit. With N entries it must have been typed in: a blank
        "(differs)" box would otherwise clear every entry on a focus-out.
        N = 1 keeps the plain rule, where _changed()'s equality check suffices."""
        if multi and not widget.isModified():
            return
        self._changed(spec, refs, widget.text())

    def _add_differs_item(self, combo, indeterminate: bool) -> None:
        """The pre-selected "(differs)" first row, whose data is None."""
        if indeterminate:
            combo.addItem(self._DIFFERS, None)

    def _build_catalog_widget(self, spec, refs, value, presentation, editable: bool) -> QWidget:
        """UnitInfo/BuildingInfo/TechInfo: a CatalogLineEdit over
        object_catalog.py's id-dataset catalog, in place of the raw spinbox.
        An empty one commits nothing, which is already the indeterminate rule."""
        indeterminate = value is trigger_fields.MIXED
        widget = CatalogLineEdit(
            presentation.catalog(),
            presentation.default_category,
            placeholder=self._DIFFERS if indeterminate else "(unset)",
            hide_internal=presentation.hide_internal,
        )
        initial = value if isinstance(value, int) and not isinstance(value, bool) else None
        widget.set_value(None if initial == trigger_fields.UNSET else initial)
        widget.setEnabled(editable)
        # Connected after set_value(), matching every other widget here: a
        # populate must not look like a user edit (trigger_fields.py's own
        # note on why _changed()'s _populating guard is otherwise unreachable).
        widget.committed.connect(lambda new, s=spec, r=refs: self._changed(s, r, new))
        return widget

    LIST_TYPE_CAPTION = "object list type"
    _LIST_TYPE_TIP = "Which kind of object the field below lists. Narrows its suggestions and Browse; not saved in the scenario."

    def _add_list_type_row(self, spec, widget: ValueLineEdit, value) -> None:
        """GH #41: the in-game Object List Type, a row above an object list
        field that narrows its picker. A filter, not a field: it never enters
        _rows, so it writes nothing and records no undo step."""
        known = widget.group_of(value) if isinstance(value, int) and not isinstance(value, bool) else ""
        # The value's own kind, else this selection's last choice, else the picker's default category.
        current = known or self._list_types.get(spec.name) or trigger_fields.CATALOG_PRESENTATIONS[spec.presentation].default_category
        combo = QComboBox()
        _fit_combo_width(combo)
        combo.addItems(object_catalog.OBJECT_CATEGORIES)
        combo.setCurrentText(current)
        combo.setEnabled(widget.isEnabled())
        combo.setToolTip(self._LIST_TYPE_TIP)
        self.property_form.addRow(self.LIST_TYPE_CAPTION, combo)
        self.property_form.labelForField(combo).setToolTip(self._LIST_TYPE_TIP)
        widget.set_group(combo.currentText())
        # Connected after the index is set, like every field widget: a populate is not a choice.
        combo.currentIndexChanged.connect(lambda _i, n=spec.name, c=combo, w=widget: self._list_type_chosen(n, c.currentText(), w))
        widget.committed.connect(lambda new, n=spec.name, c=combo, w=widget: self._list_type_follows(n, c, w, new))

    def _list_type_chosen(self, name: str, group: str, widget: ValueLineEdit) -> None:
        widget.set_group(group)
        self._list_types[name] = group

    def _list_type_follows(self, name: str, combo: QComboBox, widget: ValueLineEdit, value: int) -> None:
        """A committed value of another kind (a typed name or raw id) turns the row to its kind."""
        group = widget.group_of(value)
        if not group:
            return
        combo.blockSignals(True)
        try:
            combo.setCurrentText(group)
        finally:
            combo.blockSignals(False)
        self._list_type_chosen(name, group, widget)

    def _build_enum_picker_widget(self, spec, refs, value, editable: bool) -> QWidget:
        """A large ENUM (trigger_fields.wants_picker()): a ValueLineEdit over
        the enum's (label, value) pairs, in place of a combo too long to scroll.

        Unlike _build_catalog_widget(), UNSET (-1) is not mapped to None: an
        enum holding -1 renders "unknown (-1)", exactly as its combo did. Only
        None, an attribute _read() could not reach, shows the placeholder.
        """
        indeterminate = value is trigger_fields.MIXED
        widget = ValueLineEdit(
            [PickerItem(label, choice) for label, choice in spec.choices],
            show_values=True,
            browse_title=spec.label.capitalize(),
            placeholder=self._DIFFERS if indeterminate else "(unset)",
        )
        widget.set_value(value if isinstance(value, int) and not isinstance(value, bool) else None)
        widget.setEnabled(editable)
        # Connected after set_value(), matching every other widget here.
        widget.committed.connect(lambda new, s=spec, r=refs: self._changed(s, r, new))
        return widget

    def _build_document_reference_widget(self, spec, refs, value, editable: bool) -> QComboBox:
        """TriggerId/VariableId: a combo box over the open document's own
        trigger or variable list, resolved fresh every populate -- trigger_id
        is renumbered by remove_triggers()/reorder_triggers(), so a stale
        combo would silently point at the wrong trigger after a structural
        edit.

        No _fit_combo_width() here: that caps a combo at 12 characters, sized
        for enum labels, where a trigger or variable name runs far longer and
        would clip -- WrapLongRows already puts a wide editor on its own line
        instead of forcing the form into horizontal overflow.
        """
        indeterminate = value is trigger_fields.MIXED
        widget = QComboBox()
        self._add_differs_item(widget, indeterminate)
        widget.addItem("(unset)", trigger_fields.UNSET)
        manager = self._manager()
        if manager is not None:
            choices = (
                object_catalog.trigger_choices(manager)
                if spec.presentation == "TriggerId"
                else object_catalog.variable_choices(manager)
            )
            for label, choice in choices:
                widget.addItem(f"{label} ({choice})", choice)
        known = widget.findData(value) >= 0
        if not indeterminate and not known and isinstance(value, int) and not isinstance(value, bool):
            # An id outside the document's current list -- kept as its own
            # row rather than snapped to "(unset)", which would rewrite a
            # field the user never touched (same reasoning as the ENUM branch).
            widget.addItem(f"unknown ({value})", value)
        widget.setCurrentIndex(0 if indeterminate else max(widget.findData(value), 0))
        widget.setEnabled(editable)
        widget.currentIndexChanged.connect(lambda _, s=spec, r=refs, w=widget: self._combo_changed(s, r, w))
        return widget

    @staticmethod
    def _trigger_spinbox(value, editable: bool, unset: bool = True) -> QSpinBox:
        """_make_spinbox() with this panel's own range and unset sentinel.
        -1 is the library's unset marker, shown as text rather than as a
        number the user would have to know the meaning of. MIXED gets the
        blank indeterminate box (GH #60). `unset=False` is for a field with
        no sentinel (description_order, a u32): minimum 0, no "(unset)" text."""
        indeterminate = value is trigger_fields.MIXED
        return _make_spinbox(
            None if indeterminate else value,
            editable,
            minimum=trigger_fields.UNSET if unset else 0,
            maximum=2**31 - 1,
            special_value_text="(unset)" if unset else "",
            indeterminate=indeterminate,
        )

    @staticmethod
    def _trigger_float_spinbox(value, editable: bool) -> QDoubleSpinBox:
        """_trigger_spinbox()'s float counterpart, for quantity_float: same
        unset sentinel, same keyboard tracking off."""
        indeterminate = value is trigger_fields.MIXED
        spin = IndeterminateDoubleSpinBox() if indeterminate else QDoubleSpinBox()
        spin.setDecimals(3)
        spin.setRange(trigger_fields.UNSET, 2**31 - 1)
        spin.setSpecialValueText("(unset)")
        number = not indeterminate and isinstance(value, (int, float)) and not isinstance(value, bool)
        spin.setValue(float(value) if number else trigger_fields.UNSET)
        spin.setEnabled(editable)
        spin.setKeyboardTracking(False)
        if indeterminate:
            spin.set_indeterminate()
        return spin
