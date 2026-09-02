"""MessagesPanel: the Messages mode left page -- six scrollable text boxes
for the scenario's Instructions/Hints/Victory/Loss/History/Scouts prose.

Dumb by design, exactly as MapOptionsPanel is: it never imports
MessagesEditModel and never touches EditHistory. It reports "the user set
field X to raw value V" through the callback it is constructed with, and the
window decides what that means.
"""

from __future__ import annotations

from typing import Mapping, Sequence

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from descape.messages_fields import MESSAGE_FIELDS, STRING_ID_UNSET, normalize_for_display


class _CommitOnBlurTextEdit(QPlainTextEdit):
    """QPlainTextEdit has no built-in editingFinished -- QLineEdit's "commit
    once, when the user is done, not on every keystroke" signal.
    trigger_panel.py's QLineEdit fields all key off editingFinished for
    exactly that reason: textChanged fires per character, so wiring a
    commit straight to it would push one undo record -- and one status-log
    line -- per keystroke. This reaches the same commit point via focus
    loss instead, the nearest QPlainTextEdit equivalent."""

    editingFinished = pyqtSignal()

    def focusOutEvent(self, event) -> None:
        super().focusOutEvent(event)
        self.editingFinished.emit()


class MessagesPanel(QWidget):
    """Instructions/Hints/Victory/Loss/History/Scouts, one QGroupBox each.

    Values are raw: a text field's value is the display string (already
    normalized for a lone `\\r`/`\\n` -- see descape/messages_fields.py), an
    id field's value is the raw string-table id (STRING_ID_UNSET if unset).
    """

    MIN_USEFUL_WIDTH = 300
    # A field's whole point is holding real prose -- left at its default
    # sizeHint (about 3 lines), the box shows barely a sentence before
    # scrolling. Applied as a minimum, not a fixed height, the same reason
    # viewer.py's status_log uses setMinimumHeight over setFixedHeight: a
    # fixed height would pin the box and stop the group box (and the
    # QScrollArea's layout generally) from giving it more room when there's
    # space to spare.
    MIN_TEXT_LINES = 6

    _NO_DOCUMENT = "No map open."
    _READ_ONLY_NOTE = "Read-only for this file -- nothing here can be written back."
    _SET_ID_WARNING = (
        "The game shows language-file string #{id} for this field, not the "
        "text below."
    )
    _CLEAR_ID_LABEL = "Use my text (clear string ID)"

    def __init__(self, on_message_field=None):
        super().__init__()
        self._on_message_field = on_message_field or (lambda *args: None)

        self._loaded = None
        self._values: dict[str, str | int] = {}
        self._editable = False
        self._read_only_reasons: dict[str, str] = {}
        self._widgets: dict[str, QWidget] = {}
        self._warning_labels: dict[str, QLabel] = {}
        self._clear_id_buttons: dict[str, QPushButton] = {}
        self._populating = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.status = QLabel(self._NO_DOCUMENT)
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

        self.area = QScrollArea()
        self.area.setWidgetResizable(True)
        layout.addWidget(self.area, stretch=1)
        self._rebuild_host()

    def _rebuild_host(self) -> None:
        self.host = QWidget()
        self.host_layout = QVBoxLayout(self.host)
        self.host_layout.setContentsMargins(6, 6, 6, 6)
        self.area.setWidget(self.host)
        self._widgets = {}
        self._warning_labels = {}
        self._clear_id_buttons = {}

    # -- document state ---------------------------------------------------

    def clear_document(self) -> None:
        self._populating = True
        try:
            self._loaded = None
            self._values = {}
            self._editable = False
            self._read_only_reasons = {}
            self._rebuild_host()
            self.status.setText(self._NO_DOCUMENT)
            self.status.setToolTip("")
        finally:
            self._populating = False

    def show_scenario(
        self,
        loaded,
        values: Mapping[str, str | int] | None = None,
        editable: bool = True,
        read_only_reasons: Mapping[str, str] | None = None,
        notes: Sequence[str] = (),
    ) -> None:
        """Populate from `loaded`'s Messages section, one group box per field.

        `values` overrides individual fields' values by field_id (a text
        field id or its "<id>_id" counterpart) -- the window passes it
        whenever there are pending edits, the same reason
        MapOptionsPanel.show_scenario()'s own `values` override exists: a
        splice write never mutates the retriever it will replace, so a
        repopulate that skipped this would silently reset the form to the
        file's values while the model still held the edits.

        `editable` is a single flag, not per-field like MapOptionsPanel's --
        Messages mode has one write gate (messages_write_supported), not a
        per-row one.
        """
        if loaded is None:
            self.clear_document()
            return

        self._populating = True
        try:
            self._loaded = loaded
            retriever_map = loaded._scenario.sections["Messages"].retriever_map
            self._values = {}
            for spec in MESSAGE_FIELDS:
                text = retriever_map[spec.retriever].data
                display, _token = normalize_for_display(text) if isinstance(text, str) else (str(text), None)
                self._values[spec.field_id] = display
                self._values[f"{spec.field_id}_id"] = retriever_map[spec.id_retriever].data
            if values:
                self._values.update({k: v for k, v in values.items() if k in self._values})
            self._editable = editable
            self._read_only_reasons = dict(read_only_reasons or {})
            self._rebuild_host()
            self._build_groups()
            text, tooltip = self._status_text(loaded, notes)
            self.status.setText(text)
            self.status.setToolTip(tooltip)
        finally:
            self._populating = False

    def _status_text(self, loaded, notes: Sequence[str]) -> tuple[str, str]:
        set_ids = sum(
            1
            for spec in MESSAGE_FIELDS
            if self._values.get(f"{spec.field_id}_id", STRING_ID_UNSET) != STRING_ID_UNSET
        )
        summary = [f"6 fields — scenario {loaded.scenario_version}"]
        detail = list(notes)
        if not self._editable:
            summary.append("read-only")
            detail.insert(0, self._read_only_reasons.get("*", self._READ_ONLY_NOTE))
        if set_ids:
            summary.append(f"{set_ids} using a language-file string")
        return ", ".join(summary) + ".", " ".join(detail)

    # -- building the form --------------------------------------------------

    def _build_groups(self) -> None:
        for spec in MESSAGE_FIELDS:
            box = QGroupBox(spec.label)
            box_layout = QVBoxLayout(box)

            string_id = self._values.get(f"{spec.field_id}_id", STRING_ID_UNSET)
            if string_id != STRING_ID_UNSET:
                warning_row = QHBoxLayout()
                warning = QLabel(self._SET_ID_WARNING.format(id=string_id))
                warning.setWordWrap(True)
                warning_row.addWidget(warning, stretch=1)
                clear_button = QPushButton(self._CLEAR_ID_LABEL)
                clear_button.setEnabled(self._editable)
                clear_button.clicked.connect(lambda _=False, s=spec: self._clear_string_id(s))
                warning_row.addWidget(clear_button)
                box_layout.addLayout(warning_row)
                self._warning_labels[spec.field_id] = warning
                self._clear_id_buttons[spec.field_id] = clear_button

            editor = _CommitOnBlurTextEdit()
            editor.setPlainText(str(self._values.get(spec.field_id, "")))
            editor.setEnabled(self._editable)
            line_height = editor.fontMetrics().lineSpacing()
            editor.setMinimumHeight(line_height * self.MIN_TEXT_LINES + 12)
            tip = self._read_only_reasons.get(spec.field_id, "")
            if tip:
                editor.setToolTip(tip)
            editor.editingFinished.connect(lambda s=spec, e=editor: self._changed(s, e.toPlainText()))
            box_layout.addWidget(editor)

            self._widgets[spec.field_id] = editor
            self.host_layout.addWidget(box)
        self.host_layout.addStretch(1)

    # -- reporting an edit ----------------------------------------------------

    def _changed(self, spec, value: str) -> None:
        """The one place a widget signal becomes a reported edit -- same two
        guards as MapOptionsPanel._changed(): `_populating` covers a
        programmatic populate, and the equality check stops a re-emit of the
        already-current value (e.g. tabbing through the form) from recording
        a phantom undo step."""
        if self._populating or not self._editable:
            return
        if self._values.get(spec.field_id) == value:
            return
        self._values[spec.field_id] = value
        self._on_message_field(spec.field_id, value)

    def _clear_string_id(self, spec) -> None:
        if self._populating or not self._editable:
            return
        id_field = f"{spec.field_id}_id"
        if self._values.get(id_field) == STRING_ID_UNSET:
            return
        self._values[id_field] = STRING_ID_UNSET
        self._on_message_field(id_field, STRING_ID_UNSET)

    # -- read access, for the window and for tests ---------------------------

    def current_values(self) -> dict[str, str | int]:
        return dict(self._values)

    def widget_for(self, field_id: str) -> QWidget | None:
        if field_id in self._widgets:
            return self._widgets[field_id]
        if field_id.endswith("_id"):
            return self._clear_id_buttons.get(field_id[: -len("_id")])
        return None
