"""The square map size picker shared by File > New Map… and Map > Resize Map…
(TASK-620): the game's seven sizes plus Custom in a combo, beside a size spin.

Both halves stay enabled and follow each other: picking a size fills the
spin, typing a game size selects it in the combo, any other size shows
Custom. Non-square maps (TASK-032) would add a second spin here.

Text the spin cannot take as a size (empty, or under the minimum) makes the
picker unacceptable: `validity_changed(False)` fires, and Enter in the spin
does nothing, so no dialog acts on the size Qt would silently revert to.
"""

from __future__ import annotations

from PyQt5.QtCore import QEvent, QObject, Qt, pyqtSignal
from PyQt5.QtWidgets import QComboBox, QHBoxLayout, QSpinBox, QWidget

from descape.scenario_new import MAX_MAP_TILES, MIN_MAP_TILES, STANDARD_MAP_SIZE_NAMES, STANDARD_MAP_SIZES
from descape.spin_boxes import keep_cursor_before_suffix

CUSTOM_LABEL = "Custom"
CURRENT_MARK = ", current"
_ENTER_KEYS = (Qt.Key_Return, Qt.Key_Enter)


def size_label(tiles: int) -> str:
    return f"{STANDARD_MAP_SIZE_NAMES[tiles]} ({tiles}×{tiles})"


def invalid_size_message() -> str:
    return f"Type a size from {MIN_MAP_TILES} to {MAX_MAP_TILES}."


class MapSizePicker(QWidget):
    """`combo` and `spin` in one row. `changed(tiles)` fires once per new
    size, however it was picked. `current`, the open map's size, is marked in
    the combo. With `enter_applies`, Enter in the spin or the closed combo
    takes the typed size without passing the key on, so the dialog's default
    button stays unpressed."""

    changed = pyqtSignal(int)
    validity_changed = pyqtSignal(bool)

    def __init__(
        self, tiles: int, current: int | None = None, enter_applies: bool = False, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self._enter_applies = enter_applies
        self._syncing = False
        self._acceptable = True

        self.combo = QComboBox()
        self.combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        if current is not None and not MIN_MAP_TILES <= current <= MAX_MAP_TILES:
            current = None  # the spin cannot hold it, so no entry may select it
        sizes = sorted(set(STANDARD_MAP_SIZES) | ({current} if current is not None else set()))
        for size in sizes:
            if size in STANDARD_MAP_SIZE_NAMES:
                label = size_label(size) + (CURRENT_MARK if size == current else "")
            else:
                label = f"Current ({size}×{size})"
            self.combo.addItem(label, size)
        self.combo.addItem(CUSTOM_LABEL, None)
        self.combo.setToolTip("One of the game's map sizes, or type any size in the box beside it")

        self.spin = QSpinBox()
        self.spin.setRange(MIN_MAP_TILES, MAX_MAP_TILES)
        self.spin.setSuffix(" tiles")
        keep_cursor_before_suffix(self.spin)
        self.spin.setToolTip(
            f"Square map size, {MIN_MAP_TILES}-{MAX_MAP_TILES} tiles; non-square maps are not supported yet"
        )

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.combo)
        layout.addWidget(self.spin, 1)
        self.setFocusProxy(self.combo)

        self.set_tiles(tiles)
        self._last = self.spin.value()
        self.combo.currentIndexChanged.connect(self._on_combo)
        self.spin.valueChanged.connect(self._on_spin)
        self.spin.lineEdit().textChanged.connect(self._sync_validity)
        self.spin.editingFinished.connect(self._on_editing_finished)
        # The key reaches the line edit or the spin, depending on how focus arrived.
        self.spin.installEventFilter(self)
        self.spin.lineEdit().installEventFilter(self)
        if enter_applies:
            self.combo.installEventFilter(self)

    @property
    def tiles(self) -> int:
        return self.spin.value()

    @property
    def acceptable(self) -> bool:
        return self.spin.hasAcceptableInput()

    @property
    def is_custom(self) -> bool:
        return self.combo.currentData() is None

    def set_tiles(self, tiles: int) -> None:
        self.spin.setValue(tiles)  # emits changed through _on_spin when the size differs
        self._select_for(self.spin.value())
        self._sync_validity()

    def _select_for(self, tiles: int) -> None:
        index = self.combo.findData(tiles)
        self._syncing = True
        try:
            self.combo.setCurrentIndex(index if index >= 0 else self.combo.count() - 1)
        finally:
            self._syncing = False

    def _on_combo(self, _index: int) -> None:
        if self._syncing:
            return
        tiles = self.combo.currentData()
        if tiles is not None:
            self.spin.setValue(tiles)
            self._sync_validity()

    def _on_editing_finished(self) -> None:
        # Custom over a retyped game size (no valueChanged) turns back to that size's entry.
        self._select_for(self.spin.value())
        self._sync_validity()

    def _on_spin(self, tiles: int) -> None:
        self._select_for(tiles)
        # Qt's own Enter handling re-emits valueChanged for an unchanged size.
        if tiles != self._last:
            self._last = tiles
            self.changed.emit(tiles)

    def _sync_validity(self, *_args) -> None:
        acceptable = self.acceptable
        if acceptable != self._acceptable:
            self._acceptable = acceptable
            self.validity_changed.emit(acceptable)

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:
        if event.type() != QEvent.KeyPress or event.key() not in _ENTER_KEYS:
            return False
        if not self.acceptable:
            return True
        if self._enter_applies:
            self.spin.interpretText()
            self.spin.selectAll()
            return True
        return False
