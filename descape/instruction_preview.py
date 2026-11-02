"""GH #140: one Display Instructions effect's preview. The box itself is drawn on
the map (descape/instruction_overlay.py, TASK-550); this reads the live fields
into an InstructionView and keeps the rows the box can't show, under the form.

Observe-only. bind() connects to the property form's own row widgets and
re-reads them on every change; nothing here writes the entry or reaches
TriggerPanel._changed(). Sound is shown by name only, with no playback.
"""

from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (
    QAbstractSpinBox,
    QCheckBox,
    QComboBox,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QVBoxLayout,
    QWidget,
)

from descape import asset_source, constant_picker, message_markup, object_catalog
from descape.instruction_overlay import ICON_PX, SLOT_OFFSETS, InstructionView
from descape.value_picker import ValueLineEdit
from descape.viewer_common import pixmap_from_rgba

UNSET = -1
# Checkboxes whose stored -1 ("default") and 1 both show checked, so the entry decides until toggled.
_ENTRY_FLAGS = ("play_sound", "use_tag_color_for_icon")
_SOUND_STATE = {UNSET: "(default)", 0: "(off)", 1: "(plays)"}


def _no_focus(widget: QWidget) -> None:
    widget.setFocusPolicy(Qt.NoFocus)


def _text_label() -> QLabel:
    label = QLabel()
    label.setWordWrap(True)
    label.setTextInteractionFlags(Qt.NoTextInteraction)
    # Explicit, so one long unbreakable word cannot widen the whole form.
    label.setMinimumWidth(1)
    _no_focus(label)
    return label


class InstructionPreview(QWidget):
    """The live InstructionView, plus a note line (position and string id
    caveats) and a footer (display time and sound)."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        _no_focus(self)
        self.note_label = _text_label()
        self.footer_label = _text_label()
        column = QVBoxLayout(self)
        column.setContentsMargins(0, 4, 0, 0)
        column.setSpacing(2)
        column.addWidget(self.note_label)
        column.addWidget(self.footer_label)

        self.view: InstructionView | None = None
        self.icon_source: str | None = None
        self._fields: dict = {}
        self._entry = None
        self._read = None
        self._flags: dict[str, object] = {}
        self._variable_names: tuple[str, ...] = ()
        self._player_colors = None
        self._on_resize = None
        self._on_view = None
        self._icon_key = object()
        self._icon_pixmap: QPixmap | None = None
        self._last_height = None

    # -- wiring --------------------------------------------------------------

    def bind(
        self, fields: dict, entry, read, variable_names=(), player_colors=None, on_resize=None, on_view=None
    ) -> None:
        """Follow `fields` (field name -> its row widget) for `entry`. `read`
        is the panel's own getattr; `on_resize` runs when the preview's height
        for its width changes, so the form can re-fit; `on_view` takes each
        rebuilt InstructionView."""
        self._fields = dict(fields)
        self._entry = entry
        self._read = read
        self._flags = {name: read(entry, name) for name in _ENTRY_FLAGS}
        self._variable_names = tuple(variable_names)
        self._player_colors = player_colors
        self._on_resize = on_resize
        self._on_view = on_view
        for widget in self._fields.values():
            # Bound methods of this QObject: the connections die with it.
            if isinstance(widget, QCheckBox):
                widget.stateChanged.connect(self._flag_changed)
            elif isinstance(widget, QPlainTextEdit):
                widget.textChanged.connect(self._field_changed)
            elif isinstance(widget, QComboBox):
                widget.currentIndexChanged.connect(self._field_changed)
            elif isinstance(widget, QAbstractSpinBox):
                widget.valueChanged.connect(self._field_changed)
            elif isinstance(widget, QLineEdit):
                widget.textChanged.connect(self._field_changed)
            elif isinstance(widget, ValueLineEdit):
                widget.committed.connect(self._field_changed)
        self.refresh()

    def detach(self) -> None:
        """Stop following: the rows are being torn down with this preview."""
        self._fields = {}
        self._entry = None
        self._on_resize = None
        self._on_view = None

    def set_player_colors(self, colors) -> None:
        self._player_colors = colors
        self.refresh()

    def _field_changed(self, *_args) -> None:
        self.refresh()

    def _flag_changed(self, state: int) -> None:
        sender = self.sender()
        for name, widget in self._fields.items():
            if widget is sender and name in self._flags and state != Qt.PartiallyChecked:
                self._flags[name] = int(state == Qt.Checked)
        self.refresh()

    def _value(self, name: str):
        """The field's live widget value, or the entry's for a row with no
        live editor (a locked field's label, a "(differs)" choice)."""
        widget = self._fields.get(name)
        value = None
        if isinstance(widget, QPlainTextEdit):
            value = widget.toPlainText()
        elif isinstance(widget, QComboBox):
            value = widget.currentData()
        elif isinstance(widget, QAbstractSpinBox):
            value = widget.value()
        elif isinstance(widget, QLineEdit):
            value = widget.text()
        elif isinstance(widget, ValueLineEdit):
            value = widget.value()
            value = UNSET if value is None else value
        if value is None and self._entry is not None:
            value = self._read(self._entry, name)
        return value

    @staticmethod
    def _int(value, default: int = UNSET) -> int:
        if isinstance(value, bool):
            return int(value)
        return int(value) if isinstance(value, (int, float)) else default

    # -- rendering -----------------------------------------------------------

    def refresh(self) -> None:
        if self._entry is None:
            return
        colors = message_markup.text_colors()
        message = self._value("message") or ""
        notes = []
        string_id = self._int(self._value("string_id"))
        if string_id != UNSET:
            text = asset_source.resource_string(string_id)
            if text is None:
                notes.append(f"Shows language string #{string_id} (not available), else this message.")
            else:
                message = text
                notes.append(f"Text from language string #{string_id}.")
        runs = message_markup.parse(message, self._variable_names, colors)

        position = self._int(self._value("instruction_panel_position"))
        slot = position if 0 <= position < len(SLOT_OFFSETS) else 0
        if position == UNSET:
            notes.append("Position (unset, drawn as TOP).")
        elif position != slot:
            notes.append(f"Position {position} unknown, drawn as TOP.")
        self.note_label.setText("\n".join(notes))
        self.note_label.setVisible(bool(notes))

        icon = self._icon(self._int(self._value("object_list_unit_id")))
        self.view = InstructionView(message_markup.to_html(runs, colors["WHITE"]), slot, icon, self._tint(runs))

        self.footer_label.setText(f"{self._shows_for()}  |  {self._sound()}")
        self.updateGeometry()
        self._report_height()
        if self._on_view is not None:
            self._on_view(self.view)

    def _icon(self, unit_id: int) -> QPixmap | None:
        if unit_id != self._icon_key:
            self._icon_key = unit_id
            self._icon_pixmap, self.icon_source = None, None
            if unit_id >= 0:
                arr = asset_source.get_unit_icon(*object_catalog.portrait_for(unit_id))
                if arr is not None:
                    self._icon_pixmap, self.icon_source = pixmap_from_rgba(arr, ICON_PX), "icon"
                else:
                    sprite = constant_picker.preview_pixmap(unit_id)
                    if sprite is not None:
                        scaled = sprite.scaled(ICON_PX, ICON_PX, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                        self._icon_pixmap, self.icon_source = scaled, "sprite"
        return self._icon_pixmap

    def _tint(self, runs) -> tuple[int, int, int] | None:
        """The leading tag colour when use_tag_color_for_icon is 1, else the
        source player's colour; None with no player colours."""
        if self._flags.get("use_tag_color_for_icon") == 1:
            leading = message_markup.leading_color(runs)
            if leading is not None:
                return leading
        player = self._int(self._value("source_player"))
        colors = self._player_colors
        if colors is None or not 0 <= player < len(colors):
            return None
        return tuple(int(channel) for channel in colors[player][:3])

    def _shows_for(self) -> str:
        seconds = self._int(self._value("display_time"))
        return "Shows for (unset)" if seconds == UNSET else f"Shows for {seconds} s"

    def _sound(self) -> str:
        name = self._value("sound_name") or ""
        play = self._int(self._flags.get("play_sound"))
        state = _SOUND_STATE.get(play, f"(play_sound {play})")
        return f"Sound: {name} {state}" if name else f"Sound: none {state}"

    def _report_height(self) -> None:
        width = self.width()
        height = self.heightForWidth(width) if width > 0 else None
        if height != self._last_height:
            self._last_height = height
            if self._on_resize is not None and height is not None:
                self._on_resize()
