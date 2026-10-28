"""Settings > Tools: what each Tool Options control starts at (daubED's model).

One row per descape/tool_options.py option, grouped by its GROUPS entry: a
label, a control and a Default button. An edit stores the default
(settings.set_tool_option_defaults()), puts it on the toolbar row at once
(ViewerWindow.apply_tool_option_default()) and logs a status line.

The dialog is modal and rebuilt on every open, so the page reads the store
once, at construction. It fills its controls with their signals blocked and
only then connects them, so opening and closing the dialog writes nothing.

A stored id this session's game data can't show (a wall const or Cliff family
another dataset has) gets one extra entry, "Unavailable (<value>)": the value
shows, is kept, and Default still clears it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PyQt5.QtCore import QEvent, Qt
from PyQt5.QtWidgets import (
    QAction,
    QCheckBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from descape import cliff_catalog, settings, tool_option_bindings, tool_options
from descape.viewer_common import FocusWheelComboBox, KeepOpenMenu

if TYPE_CHECKING:
    from descape.viewer import ViewerWindow


class _FocusWheelSpinBox(QSpinBox):
    """FocusWheelComboBox's rule for a spin box: the wheel scrolls the page until the box has focus."""

    def __init__(self, *args) -> None:
        super().__init__(*args)
        self.setFocusPolicy(Qt.StrongFocus)

    def wheelEvent(self, event) -> None:  # Qt override
        if self.hasFocus():
            super().wheelEvent(event)
        else:
            event.ignore()


def _unavailable(value: object) -> str:
    return f"Unavailable ({value})"


# Combos are this many characters wide and the set buttons' text is elided to it, so the labels take the slack (TASK-498).
_CONTROL_CHARS = 14


class _SetButton(QToolButton):
    """Shows its list elided to the combos' text width, re-elided on a font change; the tooltip has it whole."""

    def __init__(self) -> None:
        super().__init__()
        self._full = ""

    def set_full_text(self, text: str) -> None:
        self._full = text
        self.setToolTip(text)
        self._elide()

    def _elide(self) -> None:
        metrics = self.fontMetrics()
        self.setText(metrics.elidedText(self._full, Qt.ElideRight, metrics.horizontalAdvance("X") * _CONTROL_CHARS))

    def changeEvent(self, event) -> None:  # Qt override
        super().changeEvent(event)
        if event.type() == QEvent.FontChange:
            self._elide()


class ToolDefaultsPage(QWidget):
    def __init__(self, window: ViewerWindow):
        super().__init__()
        self._window = window
        self._values = settings.get_tool_option_defaults()
        self.labels: dict[str, QLabel] = {}
        self.controls: dict[str, QWidget] = {}
        self.default_buttons: dict[str, QPushButton] = {}
        self._set_actions: dict[str, dict[object, QAction]] = {}
        # (group id, header label, [option names]) in tab order, for tests and the H pass.
        self.sections: list[tuple[str, QLabel, list[str]]] = []

        layout = QVBoxLayout(self)
        intro = QLabel(
            "Each launch starts the Tool Options row at these values. A change on the toolbar "
            "lasts until DEscape closes."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        buttons = QHBoxLayout()
        self.load_from_toolbar_button = QPushButton("Load From Toolbar")
        self.load_from_toolbar_button.setToolTip("Make every control's current toolbar value its default")
        buttons.addWidget(self.load_from_toolbar_button)
        self.reset_all_button = QPushButton("Reset All to Defaults")
        buttons.addWidget(self.reset_all_button)
        buttons.addStretch(1)
        layout.addLayout(buttons)

        rows = QWidget()
        grid = QGridLayout(rows)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setColumnStretch(0, 1)
        row = 0
        for group_id, title in tool_options.GROUPS:
            names = [spec.name for spec in tool_options.OPTIONS if spec.group == group_id]
            if row:
                divider = QFrame()
                divider.setFrameShape(QFrame.HLine)
                divider.setFrameShadow(QFrame.Sunken)
                grid.addWidget(divider, row, 0, 1, 3)
                row += 1
            header = QLabel(f"<b>{title}</b>")
            grid.addWidget(header, row, 0, 1, 3)
            row += 1
            self.sections.append((group_id, header, names))
            if group_id == "brush":
                shared = ", ".join(tool.label for tool in settings.TOOLS if tool.supports_brush)
                note = QLabel(f"<small>Shared by {shared}.</small>")
                note.setWordWrap(True)
                grid.addWidget(note, row, 0, 1, 3)
                row += 1
            for name in names:
                label = QLabel(tool_options.option(name).label)
                label.setWordWrap(True)
                self.labels[name] = label
                grid.addWidget(label, row, 0)
                control = self._make_control(name)
                self.controls[name] = control
                grid.addWidget(control, row, 1, Qt.AlignLeft)
                default_button = QPushButton("Default")
                default_button.setToolTip(f"Back to DEscape's own default: {self._shown(name, tool_options.default(name))}")
                self.default_buttons[name] = default_button
                grid.addWidget(default_button, row, 2)
                row += 1
        layout.addWidget(rows)

        closing = QLabel("Target always starts on Terrain.")
        closing.setWordWrap(True)
        layout.addWidget(closing)
        layout.addStretch(1)

        for name in self.controls:
            self._show(name, self._values[name])
        for name, control in self.controls.items():
            self._connect(name, control)
            self.default_buttons[name].clicked.connect(lambda _checked=False, n=name: self._reset_one(n))
        self.load_from_toolbar_button.clicked.connect(self._load_from_toolbar)
        self.reset_all_button.clicked.connect(self._reset_all)

    # -- controls ---------------------------------------------------------

    def _make_control(self, name: str) -> QWidget:
        spec = tool_options.option(name)
        if spec.kind == tool_options.BOOL:
            return QCheckBox()
        if spec.kind == tool_options.INT:
            spin = _FocusWheelSpinBox()
            row_widget = self._window.tool_option_bindings[name].widgets[0]
            spin.setSuffix(row_widget.suffix())
            spin.setRange(spec.minimum, spec.maximum if spec.maximum is not None else spec.minimum)
            return spin
        if spec.kind == tool_options.SET:
            button = _SetButton()
            button.setPopupMode(QToolButton.InstantPopup)
            menu = KeepOpenMenu(button)
            self._set_actions[name] = {}
            for label, member in self._entries(name):
                action = QAction(label, button, checkable=True)
                menu.addAction(action)
                self._set_actions[name][member] = action
            button.setMenu(menu)
            return button
        combo = FocusWheelComboBox()
        combo.setSizeAdjustPolicy(FocusWheelComboBox.AdjustToMinimumContentsLengthWithIcon)
        combo.setMinimumContentsLength(_CONTROL_CHARS)
        # A long entry is cut in the closed box; the tooltip shows it whole.
        combo.currentIndexChanged.connect(lambda _index, box=combo: box.setToolTip(box.currentText()))
        return combo

    def _entries(self, name: str) -> list[tuple[str, object]]:
        return self._window.tool_option_bindings[name].entries(self._values)

    def _connect(self, name: str, control: QWidget) -> None:
        spec = tool_options.option(name)
        if spec.kind == tool_options.BOOL:
            control.toggled.connect(lambda checked, n=name: self._edited(n, checked))
        elif spec.kind == tool_options.INT:
            control.valueChanged.connect(lambda value, n=name: self._edited(n, value))
        elif spec.kind == tool_options.SET:
            for action in self._set_actions[name].values():
                action.toggled.connect(lambda _checked, n=name: self._edited(n, self._ticked(n)))
        else:
            control.currentIndexChanged.connect(lambda index, n=name, box=control: self._edited(n, box.itemData(index)))

    def _ticked(self, name: str) -> list:
        return [member for member, action in self._set_actions[name].items() if action.isChecked()]

    def _show(self, name: str, value) -> None:
        """`value` on `name`'s control, signals blocked: displaying is never an edit."""
        spec = tool_options.option(name)
        control = self.controls[name]
        if spec.kind == tool_options.SET:
            for member, action in self._set_actions[name].items():
                action.blockSignals(True)
                action.setChecked(member in value)
                action.blockSignals(False)
            control.set_full_text(self._shown(name, value))
            return
        control.blockSignals(True)
        try:
            if spec.kind == tool_options.BOOL:
                control.setChecked(value)
            elif spec.kind == tool_options.INT:
                if name == "cliff_frame":
                    control.setMaximum(self._frame_max())
                control.setValue(value)
            else:
                control.clear()
                for label, data in self._entries(name):
                    control.addItem(label, data)
                index = tool_option_bindings.find_data(control, value)
                # None with no entries is an unavailable family's "first piece": the list stays empty.
                if index < 0 and value is not None:
                    control.addItem(_unavailable(value), value)
                    index = control.count() - 1
                control.setCurrentIndex(index)
                control.setToolTip(control.currentText())
        finally:
            control.blockSignals(False)

    def _frame_max(self) -> int:
        const = tool_option_bindings.cliff_piece_const(
            self._window, self._values["cliff_family"], self._values["cliff_piece"]
        )
        return max(0, cliff_catalog.frame_count(const) - 1) if const is not None else 0

    def _shown(self, name: str, value) -> str:
        """`value` in words, for the status line and the set buttons."""
        spec = tool_options.option(name)
        if spec.kind == tool_options.BOOL:
            return "on" if value else "off"
        if spec.kind == tool_options.INT:
            return f"{value}{self._window.tool_option_bindings[name].widgets[0].suffix()}"
        labels = {repr(data): label for label, data in self._entries(name)}
        if spec.kind == tool_options.SET:
            return ", ".join(labels.get(repr(member), str(member)) for member in value) or "none"
        return labels.get(repr(value), _unavailable(value))

    # -- edits ------------------------------------------------------------

    def _edited(self, name: str, value) -> None:
        if name == "cliff_family":
            # The row's family handler resets the piece but not the frame, so both go explicitly.
            self._store(name, {"cliff_family": value, "cliff_piece": None, "cliff_frame": 0})
        elif name == "cliff_piece":
            self._store(name, {"cliff_piece": value, "cliff_frame": 0})
        else:
            self._store(name, {name: value})

    def _store(self, name: str, values: dict) -> None:
        try:
            settings.set_tool_option_defaults(values)
        except ValueError as e:
            self._window._log_status(f"Tool default not saved: {e}")
            return
        self._values.update(values)
        if len(values) == 1:
            self._window.apply_tool_option_default(name, values[name])
        else:
            self._window.reset_tool_options_to(values)
        for other in values:
            if other != name or tool_options.option(name).kind == tool_options.SET:
                self._show(other, self._values[other])
        self._window._log_status(f"{tool_options.option(name).label} default: {self._shown(name, values[name])}")

    def _reset_one(self, name: str) -> None:
        """Default: the schema default onto the control and through the edit path, so it prunes the entry."""
        value = tool_options.default(name)
        self._values[name] = value
        self._show(name, value)
        self._edited(name, value)

    def _load_from_toolbar(self) -> None:
        try:
            settings.set_tool_option_defaults(self._window.tool_option_values())
        except ValueError as e:  # read() is canonical, so not expected
            self._window._log_status(f"Tool defaults not loaded: {e}")
            return
        self._refresh()
        self._window._log_status("Tool defaults loaded from the toolbar")

    def _reset_all(self) -> None:
        settings.reset_tool_option_defaults()
        self._window.reset_tool_options_to(tool_options.defaults())
        self._refresh()
        self._window._log_status("All tool defaults reset")

    def _refresh(self) -> None:
        self._values = settings.get_tool_option_defaults()
        for name in self.controls:
            self._show(name, self._values[name])
