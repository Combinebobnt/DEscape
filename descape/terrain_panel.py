"""TerrainPanel: the Terrain-mode left page (GH #56).

The terrain picker, always visible. It replaces the row-2 "Terrain type"
combo and the modal browser its "…" button used to open, the same move
UnitsPanel made for the object catalog: the vocabulary a mode uses on every
click belongs in the sidebar, not behind a dialog that covers the map and
closes on every pick.

Picker only, by decision: map statistics stay on page 0 and are not shown in
Terrain mode. The hover readout is one elided line at the bottom, so it can
never widen or heighten the page.

Dumb like every other panel here: it holds the chosen terrain and reports a
change through `terrain_changed`. Nothing in it touches EditHistory or the
map.
"""

from __future__ import annotations

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import QLabel, QSizePolicy, QVBoxLayout, QWidget

from descape import terrain_catalog
from descape.terrain_browser import SHARED_TEXTURE_NOTE, TerrainPickerView
from descape.viewer_common import FontScaledWidth


def default_terrain_id() -> int:
    """The terrain the panel starts on: the one the retired row-2 combo
    started on (TerrainId sorted by enum name, index 0), so an untouched
    Draw stroke paints what it always did."""
    return min(terrain_catalog.terrains(), key=lambda entry: entry.name).id


class TerrainPanel(QWidget):
    # Measured (tools/gen_terrain_browser_eyeball.py): the widest row, "Beach Non Navigable Wet
    # Gravel", is a 350 px column, plus ~86 px of preview and scrollbar. At 320 four rows clipped alike.
    MIN_USEFUL_WIDTH = FontScaledWidth(440)

    terrain_changed = pyqtSignal(int)

    def __init__(self, current_id: int | None = None, parent=None):
        super().__init__(parent)
        # Never None past here: an empty default_group expands every group and decodes all 85 textures.
        current_id = default_terrain_id() if current_id is None else current_id
        # True while set_terrain() drives the tree, so its current_changed is not taken for a user pick.
        self._syncing = False
        self._terrain_id: int = current_id
        self._hover_text = ""

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.view = TerrainPickerView(current_id)
        self.view.filter_edit.setPlaceholderText("Filter terrains…")
        # ClickFocus, as UnitsPanel's catalog: a focus-follows-click tree would eat MapView's arrow keys.
        self.view.tree.setFocusPolicy(Qt.ClickFocus)
        self.view.current_changed.connect(self._on_current_changed)
        layout.addWidget(self.view, stretch=1)

        self.note_label = QLabel(SHARED_TEXTURE_NOTE)
        self.note_label.setWordWrap(True)
        layout.addWidget(self.note_label)

        self.hover_label = QLabel("")
        # Ignored width: a plain QLabel's minimum is its full text width, which would floor the left column.
        self.hover_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.hover_label.setMinimumWidth(1)
        layout.addWidget(self.hover_label)

    # -- the chosen terrain --------------------------------------------------

    def terrain_id(self) -> int:
        """What the next paint uses. Sticky: a filter or a group-heading
        selection never clears it."""
        return self._terrain_id

    def _on_current_changed(self, value) -> None:
        # None is a group heading or a filtered-away row: keep the previous terrain.
        if self._syncing or value is None or value == self._terrain_id:
            return
        self._terrain_id = value
        self.terrain_changed.emit(value)

    def set_terrain(self, terrain_id: int) -> bool:
        """Selects `terrain_id`, revealing it first if the show-hidden box or
        a filter has it out of view. Returns False, changing nothing, for an
        id the catalog doesn't list (the Eyedropper's "unchanged" case).

        Emits terrain_changed once if the terrain actually changed, like a
        user pick, so the toolbar's auto-beach gate tracks it either way.
        """
        entry = next((e for e in terrain_catalog.terrains() if e.id == terrain_id), None)
        if entry is None:
            return False
        self._syncing = True
        try:
            if entry.hidden and self.view.show_hidden_checkbox is not None:
                self.view.show_hidden_checkbox.setChecked(True)
            self.view.select(terrain_id)
            if self.view.current_value() != terrain_id:
                # Only a filter can still hide the row here, so clear it rather than select an invisible row.
                self.view.filter_edit.setText("")
                self.view.select(terrain_id)
        finally:
            self._syncing = False
        if terrain_id != self._terrain_id:
            self._terrain_id = terrain_id
            self.terrain_changed.emit(terrain_id)
        return True

    # -- hover readout -------------------------------------------------------

    def set_hover_text(self, text: str) -> None:
        self._hover_text = text
        self._apply_hover_text()

    def hover_text(self) -> str:
        """The full, unelided readout (the label itself shows it elided)."""
        return self._hover_text

    def _apply_hover_text(self) -> None:
        # Elided, not wrapped: wrapping would grow the page by a row, and a QLabel clips mid-glyph.
        width = max(self.hover_label.width(), 1)
        metrics = self.hover_label.fontMetrics()
        self.hover_label.setText(metrics.elidedText(self._hover_text, Qt.ElideRight, width))
        self.hover_label.setToolTip(self._hover_text)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._apply_hover_text()
