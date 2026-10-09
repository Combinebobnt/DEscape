"""File > Resize Map…: the size + anchor picker, with a live summary read
from scenario_resize.resize_plan(), so what the dialog says is exactly what
the resize will do (how many objects go, whether trigger areas follow).

Square-only for now: "Keep square" is ticked and locked, so the height spin
follows the width spin. Non-square maps wait on the in-game check TASK-032's
Phase 0 owes; unlocking it is the only change this dialog needs then.
"""

from __future__ import annotations

from collections.abc import Callable

from PyQt5.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QLabel,
    QRadioButton,
    QSpinBox,
    QVBoxLayout,
)

from descape.scenario_new import MAX_MAP_TILES, MIN_MAP_TILES
from descape.scenario_resize import Anchor, ResizePlan

_ANCHOR_TIPS = {
    Anchor.TOP_LEFT: "Keep the tile (0, 0) corner: the west tip in the isometric view",
    Anchor.BOTTOM_RIGHT: "Keep the far corner: the east tip in the isometric view",
    Anchor.CENTER: "Keep the middle of the map; an odd change biases toward tile (0, 0)",
}


class ResizeDialog(QDialog):
    """Width, height, a locked "Keep square" box, a 3x3 anchor grid and the
    plan summary. `plan_for(width, height, anchor)` is called on every
    change; OK is enabled only while the plan has no refusal."""

    def __init__(
        self,
        parent,
        width: int,
        height: int,
        plan_for: Callable[[int, int, Anchor], ResizePlan],
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Resize Map")
        self._plan_for = plan_for
        self.plan: ResizePlan | None = None

        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.width_spin = QSpinBox()
        self.height_spin = QSpinBox()
        for spin, value in ((self.width_spin, width), (self.height_spin, height)):
            spin.setRange(MIN_MAP_TILES, MAX_MAP_TILES)
            spin.setValue(value)
            spin.setSuffix(" tiles")
        self.width_spin.setToolTip("New size along x (the map's upper-left and lower-right edges)")
        self.height_spin.setToolTip("New size along y (the map's lower-left and upper-right edges)")
        form.addRow("Width:", self.width_spin)
        form.addRow("Height:", self.height_spin)
        self.square_box = QCheckBox("Keep square")
        self.square_box.setChecked(True)
        self.square_box.setEnabled(False)
        self.square_box.setToolTip(
            "Non-square maps are not supported yet: whether AoE2:DE accepts them is still "
            "being checked in-game"
        )
        form.addRow("", self.square_box)
        layout.addLayout(form)

        anchor_box = QGroupBox("Anchor (the part of the map that stays put)")
        grid = QGridLayout(anchor_box)
        self.anchor_group = QButtonGroup(self)
        self.anchor_buttons: dict[Anchor, QRadioButton] = {}
        for anchor in Anchor:
            column, row = anchor.value
            button = QRadioButton(anchor.label.capitalize())
            button.setToolTip(_ANCHOR_TIPS.get(anchor, f"Keep the map's {anchor.label} in place"))
            self.anchor_group.addButton(button)
            self.anchor_buttons[anchor] = button
            grid.addWidget(button, row, column)
        self.anchor_buttons[Anchor.TOP_LEFT].setChecked(True)
        layout.addWidget(anchor_box)

        self.summary_label = QLabel()
        self.summary_label.setWordWrap(True)
        layout.addWidget(self.summary_label)

        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.button(QDialogButtonBox.Ok).setText("Resize")
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)

        self.width_spin.valueChanged.connect(self._on_width)
        self.height_spin.valueChanged.connect(self._on_height)
        self.anchor_group.buttonToggled.connect(lambda _button, checked: checked and self._refresh())
        self._refresh()

    def _on_width(self, value: int) -> None:
        if self.square_box.isChecked() and self.height_spin.value() != value:
            self.height_spin.setValue(value)  # refreshes through _on_height
            return
        self._refresh()

    def _on_height(self, value: int) -> None:
        if self.square_box.isChecked() and self.width_spin.value() != value:
            self.width_spin.setValue(value)
            return
        self._refresh()

    @property
    def anchor(self) -> Anchor:
        return next(a for a, b in self.anchor_buttons.items() if b.isChecked())

    @property
    def new_size(self) -> tuple[int, int]:
        return self.width_spin.value(), self.height_spin.value()

    def _refresh(self) -> None:
        width, height = self.new_size
        self.plan = self._plan_for(width, height, self.anchor)
        self.summary_label.setText(self.plan.summary())
        self.buttons.button(QDialogButtonBox.Ok).setEnabled(self.plan.refusal is None)
