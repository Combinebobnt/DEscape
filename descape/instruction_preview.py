"""GH #140: a mock game screen previewing one Display Instructions effect.

Observe-only. bind() connects to the property form's own row widgets and
re-reads them on every change; nothing here writes the entry or reaches
TriggerPanel._changed(). Sound is shown by name only, with no playback.
"""

from __future__ import annotations

import numpy as np
from PyQt5.QtCore import QSize, Qt
from PyQt5.QtGui import QColor, QImage, QPalette, QPixmap
from PyQt5.QtWidgets import (
    QAbstractSpinBox,
    QCheckBox,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from descape import asset_source, constant_picker, message_markup, object_catalog
from descape.value_picker import ValueLineEdit

UNSET = -1
POSITIONS = ("TOP", "MIDDLE", "BOTTOM")
ICON_PX = 48
_ICON_BORDER = 2
_SCREEN_ASPECT = 9 / 16
_SCREEN_MARGIN = 6
_SCREEN_COLOR = QColor(24, 28, 22)
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


def _pixmap_from_rgba(arr) -> QPixmap:
    arr = np.ascontiguousarray(arr)
    height, width = arr.shape[:2]
    image = QImage(arr.data, width, height, 4 * width, QImage.Format_RGBA8888)
    # fromImage copies, so the array may go out of scope.
    return QPixmap.fromImage(image).scaled(ICON_PX, ICON_PX, Qt.KeepAspectRatio, Qt.SmoothTransformation)


class _Screen(QWidget):
    """The dark game screen. The box is placed by hand in its third, and the
    screen grows past its aspect height when the box needs more than a third."""

    def __init__(self, box: QWidget) -> None:
        super().__init__()
        self._box = box
        self._slot = 0
        box.setParent(self)
        policy = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        policy.setHeightForWidth(True)
        self.setSizePolicy(policy)
        self.setAutoFillBackground(True)
        palette = self.palette()
        palette.setColor(QPalette.Window, _SCREEN_COLOR)
        self.setPalette(palette)
        _no_focus(self)

    def set_slot(self, slot: int) -> None:
        self._slot = slot
        self.place()

    def _box_height(self, width: int) -> int:
        inner = max(1, width - 2 * _SCREEN_MARGIN)
        box = self._box
        height = box.heightForWidth(inner) if box.hasHeightForWidth() else box.sizeHint().height()
        return max(height, box.minimumSizeHint().height())

    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, width: int) -> int:
        return max(round(width * _SCREEN_ASPECT), 3 * self._box_height(width))

    def sizeHint(self) -> QSize:
        return QSize(320, self.heightForWidth(320))

    def minimumSizeHint(self) -> QSize:
        return QSize(160, 90)

    def place(self) -> None:
        width, height = self.width(), self.height()
        if width <= 0 or height <= 0:
            return
        third = height / 3
        box_height = min(self._box_height(width), int(third))
        top = int(self._slot * third + (third - box_height) / 2)
        self._box.setGeometry(_SCREEN_MARGIN, top, width - 2 * _SCREEN_MARGIN, box_height)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self.place()


class InstructionPreview(QWidget):
    """The screen with its instruction box, a note line (position and string
    id caveats) and a footer (display time and sound)."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        _no_focus(self)
        self.box = QFrame()
        self.box.setObjectName("instructionBox")
        self.box.setStyleSheet(
            "#instructionBox { background: rgba(0, 0, 0, 170); border: 1px solid rgb(150, 130, 80); }"
        )
        _no_focus(self.box)
        row = QHBoxLayout(self.box)
        row.setContentsMargins(4, 4, 4, 4)
        row.setSpacing(6)
        self.icon_label = QLabel()
        self.icon_label.setFixedSize(ICON_PX + 2 * _ICON_BORDER, ICON_PX + 2 * _ICON_BORDER)
        self.icon_label.setAlignment(Qt.AlignCenter)
        _no_focus(self.icon_label)
        row.addWidget(self.icon_label, 0, Qt.AlignTop)
        self.text_label = _text_label()
        self.text_label.setTextFormat(Qt.RichText)
        self.text_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        row.addWidget(self.text_label, 1)

        self.screen = _Screen(self.box)
        self.note_label = _text_label()
        self.footer_label = _text_label()
        column = QVBoxLayout(self)
        column.setContentsMargins(0, 4, 0, 0)
        column.setSpacing(2)
        column.addWidget(self.screen)
        column.addWidget(self.note_label)
        column.addWidget(self.footer_label)

        self.drawn_position = 0
        self.icon_source: str | None = None
        self.tint: tuple[int, int, int] | None = None
        self._fields: dict = {}
        self._entry = None
        self._read = None
        self._flags: dict[str, object] = {}
        self._variable_names: tuple[str, ...] = ()
        self._player_colors = None
        self._on_resize = None
        self._icon_key = object()
        self._icon_pixmap: QPixmap | None = None
        self._last_height = None

    # -- wiring --------------------------------------------------------------

    def bind(self, fields: dict, entry, read, variable_names=(), player_colors=None, on_resize=None) -> None:
        """Follow `fields` (field name -> its row widget) for `entry`. `read`
        is the panel's own getattr; `on_resize` runs when the preview's height
        for its width changes, so the form can re-fit."""
        self._fields = dict(fields)
        self._entry = entry
        self._read = read
        self._flags = {name: read(entry, name) for name in _ENTRY_FLAGS}
        self._variable_names = tuple(variable_names)
        self._player_colors = player_colors
        self._on_resize = on_resize
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
        self.text_label.setText(message_markup.to_html(runs, colors["WHITE"]))

        position = self._int(self._value("instruction_panel_position"))
        self.drawn_position = position if 0 <= position < len(POSITIONS) else 0
        if position == UNSET:
            notes.append("Position (unset, drawn as TOP).")
        elif position != self.drawn_position:
            notes.append(f"Position {position} unknown, drawn as TOP.")
        self.note_label.setText("\n".join(notes))
        self.note_label.setVisible(bool(notes))

        self._show_icon(self._int(self._value("object_list_unit_id")))
        self.tint = self._tint(runs)
        border = "transparent" if self.tint is None else "#{:02x}{:02x}{:02x}".format(*self.tint)
        self.icon_label.setStyleSheet(f"border: {_ICON_BORDER}px solid {border};")

        self.footer_label.setText(f"{self._shows_for()}  |  {self._sound()}")
        self.screen.set_slot(self.drawn_position)
        self.screen.updateGeometry()
        self.updateGeometry()
        self._report_height()

    def _show_icon(self, unit_id: int) -> None:
        if unit_id != self._icon_key:
            self._icon_key = unit_id
            self._icon_pixmap, self.icon_source = None, None
            if unit_id >= 0:
                arr = asset_source.get_unit_icon(object_catalog.icon_for(unit_id))
                if arr is not None:
                    self._icon_pixmap, self.icon_source = _pixmap_from_rgba(arr), "icon"
                else:
                    sprite = constant_picker.preview_pixmap(unit_id)
                    if sprite is not None:
                        scaled = sprite.scaled(ICON_PX, ICON_PX, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                        self._icon_pixmap, self.icon_source = scaled, "sprite"
        if self._icon_pixmap is None:
            self.icon_label.clear()
        else:
            self.icon_label.setPixmap(self._icon_pixmap)
        self.icon_label.setVisible(self._icon_pixmap is not None)

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
