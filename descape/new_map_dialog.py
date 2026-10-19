"""File > New Map… (GH #243): the size, game data, terrain and elevation
picker. The dialog only collects a NewMapChoice; ViewerWindow.new_map_dialog()
asks the large-map question and generates the map.

It is drawn inside the DEscape window, over a dimmed backdrop
(InWindowModal), not as a separate window: on Wayland, and under WSLg even
in X11 mode, the desktop alone places a new window and ignores the app's
move(), so a separate window could open in a screen corner. A child widget
is placed by DEscape itself, so it is centered on every OS and session.

Custom stays square: non-square maps do not load through the pinned library
yet (see scenario_new.blank_body()). Game data picks the donor: Age of
Empires II or Return of Rome (GH #67). Both games list the same terrains, so
the Terrain list does not change with it.
"""

from __future__ import annotations

import contextlib
from dataclasses import asdict, dataclass

from PyQt5 import sip
from PyQt5.QtCore import QEvent, QEventLoop, QObject, QSize, Qt, QTimer
from PyQt5.QtGui import QColor, QFont, QIcon, QPainter
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from descape import asset_source, game_dataset, iso_geometry, settings, terrain_browser, terrain_catalog
from descape.scenario_io import BLANK_TEMPLATE_TILES
from descape.scenario_new import (
    MAX_MAP_TILES,
    MIN_MAP_TILES,
    STANDARD_MAP_SIZE_NAMES,
    STANDARD_MAP_SIZES,
)

AOE2, ROR = game_dataset.AOE2, game_dataset.ROR
CUSTOM_LABEL = "Custom"
# Terrain swatches decoded per timer tick after the dialog shows: a .dds
# decode is ~28 ms (terrain_browser), so ~112 at once held the dialog back,
# and one per tick keeps every pause to a single decode.
SWATCHES_PER_TICK = 1
# The panel is fixed at its contents' size, and never smaller than this.
MIN_PANEL_SIZE = QSize(420, 260)
_BACKDROP = QColor(0, 0, 0, 110)


@dataclass(frozen=True)
class NewMapChoice:
    tiles: int = BLANK_TEMPLATE_TILES
    game_data: str = AOE2
    terrain_id: int = settings.NEW_MAP_DEFAULT_TERRAIN_ID
    elevation: int = settings.NEW_MAP_DEFAULT_ELEVATION

    @classmethod
    def from_dict(cls, raw: dict) -> NewMapChoice:
        return cls(**{key: raw[key] for key in asdict(cls()) if key in raw})

    def as_dict(self) -> dict:
        return asdict(self)


def size_label(tiles: int) -> str:
    return f"{STANDARD_MAP_SIZE_NAMES[tiles]} ({tiles}×{tiles})"


def terrain_entries() -> list[terrain_catalog.TerrainEntry]:
    """The Terrain panel's list: catalog rows without the hidden ones, by display name."""
    entries = [entry for entry in terrain_catalog.terrains() if not entry.hidden]
    return sorted(entries, key=lambda entry: terrain_catalog.display_name(entry.name))


class NewMapDialog(QDialog):
    """Size (the seven presets, then Custom with a square size spin), Game
    data, Terrain and Elevation. After accept, `choice` holds the picked
    NewMapChoice. An initial value the widgets cannot show (an unknown
    terrain, an unknown game data, an out-of-range number) falls back to
    NewMapChoice's default for that field.

    Built once per window and reused (set_initial() before each open), so
    its swatches are decoded once; `built_for` says which game data and
    install they came from, and a change there needs a new dialog."""

    def __init__(self, parent, initial: NewMapChoice | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("New Map")
        self.built_for = self.build_key()
        self.choice: NewMapChoice | None = None

        layout = QVBoxLayout(self)
        # In the window there is no title bar, so the title is a label.
        title = QLabel("New Map")
        font = QFont(title.font())
        font.setBold(True)
        font.setPointSizeF(font.pointSizeF() * 1.2)
        title.setFont(font)
        layout.addWidget(title)
        form = QFormLayout()

        self.size_combo = QComboBox()
        for tiles in STANDARD_MAP_SIZES:
            self.size_combo.addItem(size_label(tiles), tiles)
        self.size_combo.addItem(CUSTOM_LABEL, None)
        self.size_combo.setToolTip("One of the game's map sizes, or Custom to type a size")
        self.custom_spin = QSpinBox()
        self.custom_spin.setRange(MIN_MAP_TILES, MAX_MAP_TILES)
        self.custom_spin.setSuffix(" tiles")
        self.custom_spin.setToolTip(
            f"Square map size, {MIN_MAP_TILES}-{MAX_MAP_TILES} tiles; non-square maps are not supported yet"
        )
        form.addRow("Size:", self.size_combo)
        form.addRow(f"Custom size ({MIN_MAP_TILES}-{MAX_MAP_TILES}):", self.custom_spin)

        self.game_data_combo = QComboBox()
        for game_data in settings.NEW_MAP_GAME_DATA:
            self.game_data_combo.addItem(game_dataset.DISPLAY_NAMES[game_data], game_data)
        self.game_data_combo.setToolTip("Which game the new map is for: Age of Empires II or Return of Rome")
        form.addRow("Game data:", self.game_data_combo)

        self.terrain_combo = QComboBox()
        self.terrain_combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        for entry in terrain_entries():
            self.terrain_combo.addItem(terrain_catalog.display_name(entry.name), entry.id)
        self.terrain_combo.setToolTip("The terrain every tile starts with")
        form.addRow("Terrain:", self.terrain_combo)

        self.elevation_spin = QSpinBox()
        self.elevation_spin.setRange(iso_geometry.MIN_ELEVATION, iso_geometry.MAX_ELEVATION)
        # Measured in the 1.59 editor: a stored 3 shows as height 4 there.
        self.elevation_spin.setToolTip(
            "The height every tile starts at, as stored in the file.\n"
            "The game's Scenario Editor shows it one higher (0 shows as 1)."
        )
        form.addRow("Elevation:", self.elevation_spin)
        layout.addLayout(form)

        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.button(QDialogButtonBox.Ok).setText("Create")
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)

        self.size_combo.currentIndexChanged.connect(self._sync_custom_spin)
        # Every swatch waits for the timer, except the shown one (set_initial()).
        self._pending_swatches = list(range(self.terrain_combo.count()))
        self._swatch_timer = QTimer(self)
        self._swatch_timer.setInterval(0)
        self._swatch_timer.timeout.connect(self._fill_swatches)
        self.set_initial(initial or NewMapChoice())
        self._fit_to_contents()

    @staticmethod
    def build_key() -> tuple:
        """What a built panel depends on: the active game data and the install
        (the swatch art) and the UI font (the title's size, set at build)."""
        return game_dataset.active(), asset_source.get_install_path(), QApplication.font().toString()

    def set_initial(self, initial: NewMapChoice) -> None:
        """Shows `initial` in the widgets, each field falling back to the
        default when it cannot be shown, and clears `choice`."""
        default = NewMapChoice()
        self.choice = None
        self._set_size(initial.tiles if MIN_MAP_TILES <= initial.tiles <= MAX_MAP_TILES else default.tiles)
        game_data = initial.game_data if initial.game_data in settings.NEW_MAP_GAME_DATA else default.game_data
        self.game_data_combo.setCurrentIndex(self.game_data_combo.findData(game_data))
        terrain_index = self.terrain_combo.findData(initial.terrain_id)
        if terrain_index < 0:
            terrain_index = self.terrain_combo.findData(default.terrain_id)
        self.terrain_combo.setCurrentIndex(terrain_index)
        if terrain_index in self._pending_swatches:
            self._pending_swatches.remove(terrain_index)
            self._set_swatch(terrain_index)
        in_range = iso_geometry.MIN_ELEVATION <= initial.elevation <= iso_geometry.MAX_ELEVATION
        self.elevation_spin.setValue(initial.elevation if in_range else default.elevation)

    def _set_size(self, tiles: int) -> None:
        index = self.size_combo.findData(tiles) if tiles in STANDARD_MAP_SIZES else -1
        if index < 0:
            index = self.size_combo.count() - 1
        self.custom_spin.setValue(tiles)
        self.size_combo.setCurrentIndex(index)
        self._sync_custom_spin()

    def _sync_custom_spin(self) -> None:
        self.custom_spin.setEnabled(self.is_custom)

    @property
    def is_custom(self) -> bool:
        return self.size_combo.currentData() is None

    @property
    def tiles(self) -> int:
        return self.custom_spin.value() if self.is_custom else self.size_combo.currentData()

    def current_choice(self) -> NewMapChoice:
        return NewMapChoice(
            tiles=self.tiles,
            game_data=self.game_data_combo.currentData(),
            terrain_id=self.terrain_combo.currentData(),
            elevation=self.elevation_spin.value(),
        )

    def accept(self) -> None:
        self.choice = self.current_choice()
        super().accept()

    def _set_swatch(self, index: int) -> None:
        self.terrain_combo.setItemIcon(index, QIcon(terrain_browser.swatch_pixmap(self.terrain_combo.itemData(index))))

    def _fill_swatches(self) -> None:
        batch, self._pending_swatches = (
            self._pending_swatches[:SWATCHES_PER_TICK],
            self._pending_swatches[SWATCHES_PER_TICK:],
        )
        for index in batch:
            self._set_swatch(index)
        if not self._pending_swatches:
            self._swatch_timer.stop()
            # The icons widen the Terrain dropdown, so the panel grows with them.
            self._fit_to_contents()

    def _fit_to_contents(self) -> None:
        self.setFixedSize(self.sizeHint().expandedTo(MIN_PANEL_SIZE))

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if self._pending_swatches:
            self._swatch_timer.start()

    def hideEvent(self, event) -> None:
        self._swatch_timer.stop()
        super().hideEvent(event)

    def exec_in(self, window: QWidget) -> int:
        """Runs the dialog inside `window`, centered over a dimmed backdrop,
        until Create or Cancel; returns QDialog.Accepted or Rejected. Used in
        place of exec_(), which would open a separate window."""
        self._fit_to_contents()
        modal = InWindowModal(window, self)
        loop = QEventLoop()
        self.finished.connect(loop.quit)
        # Never left spinning if the backdrop goes without a Create or Cancel.
        modal.destroyed.connect(loop.quit)
        self.setResult(QDialog.Rejected)
        try:
            modal.open()
            loop.exec_()
        finally:
            with contextlib.suppress(TypeError, RuntimeError):
                self.finished.disconnect(loop.quit)
            if not sip.isdeleted(modal):
                modal.close()
        return self.result()


_DRAG_EVENTS = (QEvent.DragEnter, QEvent.DragMove, QEvent.Drop)
_INPUT_EVENTS = (QEvent.KeyPress, QEvent.KeyRelease)


class InWindowModal(QWidget):
    """A dimmed backdrop over the whole of `window` (menu bar and toolbars
    included) with `panel` centered on it, kept centered as either resizes.
    While open it stands in for a modal window:

    - Clicks and wheel turns behind the panel land on the backdrop.
    - An application event filter swallows shortcuts and key presses aimed
      outside the panel, drags and drops anywhere, and the window's close
      request. Inside the panel a key without Ctrl, Alt or Meta overrides any
      window shortcut, so type-ahead, Esc and Enter reach the panel.
    - The menu bar's menus are disabled (macOS's native one is outside the
      window), and so is every other open DEscape window.
    - Qt.SubWindow keeps Tab focus inside the panel, and the focus goes back
      where it was on close."""

    def __init__(self, window: QWidget, panel: QWidget) -> None:
        super().__init__(window)
        self._window = window
        self._panel = panel
        self._menu_states: list = []
        self._disabled_windows: list[QWidget] = []
        self._focus_before: QWidget | None = None
        self.setAttribute(Qt.WA_NoSystemBackground, True)
        self._recenter_timer = QTimer(self)
        self._recenter_timer.setSingleShot(True)
        self._recenter_timer.timeout.connect(self._recenter)
        panel.setParent(self, Qt.SubWindow)
        panel.setAutoFillBackground(True)
        panel.installEventFilter(self)
        window.installEventFilter(self)

    def open(self) -> None:
        self._focus_before = QApplication.focusWidget()
        menu_bar = getattr(self._window, "menuBar", None)
        if callable(menu_bar):
            self._menu_states = [(action, action.isEnabled()) for action in menu_bar().actions()]
            for action, _enabled in self._menu_states:
                action.setEnabled(False)
        self._disabled_windows = [
            other
            for other in QApplication.topLevelWidgets()
            if other is not self._window
            and other.isVisible()
            and other.isEnabled()
            and other.windowType() not in (Qt.Popup, Qt.ToolTip)
        ]
        for other in self._disabled_windows:
            other.setEnabled(False)
        QApplication.instance().installEventFilter(self)
        self.setGeometry(self._window.rect())
        self.show()
        self.raise_()
        self._panel.show()
        self._recenter()
        self._panel.activateWindow()
        self._panel.size_combo.setFocus(Qt.OtherFocusReason)

    def close(self) -> bool:
        QApplication.instance().removeEventFilter(self)
        self._window.removeEventFilter(self)
        self._panel.removeEventFilter(self)
        self._recenter_timer.stop()
        # Back to the window, hidden, so deleting the backdrop keeps the panel for its next open.
        self._panel.hide()
        self._panel.setParent(self._window, Qt.SubWindow)
        for action, enabled in self._menu_states:
            action.setEnabled(enabled)
        self._menu_states = []
        for other in self._disabled_windows:
            if not sip.isdeleted(other):
                other.setEnabled(True)
        self._disabled_windows = []
        self.hide()
        before = self._focus_before
        if before is not None and not sip.isdeleted(before) and before.isVisible() and before.isEnabled():
            before.setFocus(Qt.OtherFocusReason)
        else:
            self._window.setFocus(Qt.OtherFocusReason)
        self.deleteLater()
        return True

    def _recenter(self) -> None:
        rect = self._panel.geometry()
        rect.moveCenter(self.rect().center())
        self._panel.move(rect.topLeft())
        self.update()

    def _inside_panel(self, obj: QObject) -> bool:
        return isinstance(obj, QWidget) and (obj is self._panel or self._panel.isAncestorOf(obj))

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:
        kind = event.type()
        if kind in _DRAG_EVENTS:
            event.ignore()
            return True
        if obj is self._window:
            if kind == QEvent.Resize:
                self.setGeometry(self._window.rect())
                self._recenter()
            elif kind == QEvent.Close:
                event.ignore()  # a modal window would block the close button too
                return True
            return False
        if obj is self._panel:
            if kind == QEvent.Resize:
                self._recenter_timer.start(0)
            return False
        if kind == QEvent.ShortcutOverride and self._inside_panel(obj):
            if not event.modifiers() & (Qt.ControlModifier | Qt.AltModifier | Qt.MetaModifier):
                event.accept()
                return True
            return False
        if kind == QEvent.Shortcut:
            return not self._inside_panel(obj)
        if kind in _INPUT_EVENTS and isinstance(obj, QWidget):
            return obj.window() is self._window and not self._inside_panel(obj)
        return False

    def mousePressEvent(self, event) -> None:
        event.accept()

    def mouseReleaseEvent(self, event) -> None:
        event.accept()

    def mouseDoubleClickEvent(self, event) -> None:
        event.accept()

    def wheelEvent(self, event) -> None:
        event.accept()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), _BACKDROP)
        painter.setPen(self.palette().color(self.palette().Mid))
        painter.drawRect(self._panel.geometry().adjusted(-1, -1, 0, 0))
