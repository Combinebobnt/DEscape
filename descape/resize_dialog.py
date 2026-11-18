"""Map > Resize Map…: the size + anchor picker, with a live summary read
from scenario_resize.resize_plan(), so what the dialog says is exactly what
the resize will do (how many objects go, whether trigger areas follow).
Given a PreviewSource, it also shows where, in a resize_preview.ResizePreview
beside the controls, drawn from that same plan.

Square-only for now: one Size row (map_size_picker.MapSizePicker, shared
with File > New Map…) sets both axes, and the plan refuses a non-square map.
Non-square maps wait on the in-game check TASK-032's Phase 0 owes. Enter in
the size box shows the typed size's plan without resizing, since a resize
cannot be undone.

The anchor is session-sticky (`last=` / `state()`, TASK-514); the size
always opens at the open map's.
"""

from __future__ import annotations

from collections.abc import Callable

from PyQt5.QtWidgets import (
    QButtonGroup,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QRadioButton,
    QVBoxLayout,
)

from descape.map_size_picker import MapSizePicker, invalid_size_message
from descape.resize_preview import PREVIEW_WIDTH, PreviewSource, ResizePreview
from descape.scenario_resize import Anchor, ResizePlan

_ANCHOR_TIPS = {
    Anchor.TOP_LEFT: "Keep the tile (0, 0) corner: the west tip in the isometric view",
    Anchor.BOTTOM_RIGHT: "Keep the far corner: the east tip in the isometric view",
    Anchor.CENTER: "Keep the middle of the map; an odd change biases toward tile (0, 0)",
}
_LEGEND = (
    "Pale: new ground. Dark: cut off, with everything on it. "
    "Red: player objects deleted, and anything the new edge cuts through."
)


class ResizeDialog(QDialog):
    """The size picker, a 3x3 anchor grid and the plan summary, plus the
    preview when `preview` is given. `plan_for(width, height, anchor)` is
    called on every new size or anchor; OK is enabled only while the typed
    size is a valid one and the plan has no refusal."""

    def __init__(
        self,
        parent,
        width: int,
        height: int,
        plan_for: Callable[[int, int, Anchor], ResizePlan],
        last: dict | None = None,
        preview: PreviewSource | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Resize Map")
        self._plan_for = plan_for
        self.plan: ResizePlan | None = None
        self._plan_key: tuple | None = None
        self.preview: ResizePreview | None = None

        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.size_picker = MapSizePicker(width, current=width if width == height else None, enter_applies=True)
        form.addRow("Size:", self.size_picker)

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
        self.anchor_buttons[(last or {}).get("anchor", Anchor.TOP_LEFT)].setChecked(True)

        if preview is None:
            layout.addLayout(form)
            layout.addWidget(anchor_box)
        else:
            self.preview = ResizePreview(preview)
            self.legend_label = QLabel(_LEGEND)
            self.legend_label.setWordWrap(True)
            self.legend_label.setFixedWidth(PREVIEW_WIDTH)
            controls = QVBoxLayout()
            controls.addLayout(form)
            controls.addWidget(anchor_box)
            controls.addStretch(1)
            picture = QVBoxLayout()
            picture.addWidget(self.preview)
            picture.addWidget(self.legend_label)
            picture.addStretch(1)
            columns = QHBoxLayout()
            columns.addLayout(controls)
            columns.addLayout(picture)
            layout.addLayout(columns)

        self.summary_label = QLabel()
        self.summary_label.setWordWrap(True)
        layout.addWidget(self.summary_label)

        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.button(QDialogButtonBox.Ok).setText("Resize")
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)

        self.size_picker.changed.connect(lambda _tiles: self._refresh())
        self.size_picker.validity_changed.connect(lambda _ok: self._refresh())
        self.anchor_group.buttonToggled.connect(lambda _button, checked: checked and self._refresh())
        self._refresh()

    @property
    def anchor(self) -> Anchor:
        return next(a for a, b in self.anchor_buttons.items() if b.isChecked())

    @property
    def new_size(self) -> tuple[int, int]:
        tiles = self.size_picker.tiles
        return tiles, tiles

    def state(self) -> dict:
        """Session-sticky values, restored by the next dialog. Nothing here
        is persisted to settings, so there is no MEMOIZED_GLOBALS key."""
        return {"anchor": self.anchor}

    def _refresh(self) -> None:
        ok_button = self.buttons.button(QDialogButtonBox.Ok)
        if not self.size_picker.acceptable:
            # The preview keeps the last valid size's plan until the text is one again.
            self.summary_label.setText(invalid_size_message())
            ok_button.setEnabled(False)
        else:
            key = (*self.new_size, self.anchor)
            if key != self._plan_key:
                self.plan = self._plan_for(*key)
                self._plan_key = key
                if self.preview is not None:
                    self.preview.set_plan(self.plan)
            self.summary_label.setText(self.plan.summary())
            ok_button.setEnabled(self.plan.refusal is None)
        # A longer summary wraps to more rows: grow to fit it, never shrink under the user.
        needed = self.heightForWidth(self.width())
        if self.isVisible() and needed > self.height():
            self.resize(self.width(), needed)
