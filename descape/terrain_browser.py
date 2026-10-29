"""The visual terrain picker: a category-grouped tree of terrain swatches,
living in the Terrain-mode sidebar page (descape/terrain_panel.py).

A thin adapter over descape/value_picker.py, the same shape
constant_picker.py's CatalogBrowseDialog is. (The plan for this called for a
hand-written sibling instead, because CatalogBrowseDialog was hard-wired to
object_catalog/unit_sprites at the time -- the value_picker extraction has
since lifted exactly that coupling out, so there is nothing left to copy.)

What this module adds on top is the one piece of domain knowledge
value_picker.py has none of: a terrain swatch, as a per-row icon and the
grid's tile. The icons are what make this a browser rather than a second
list of names, so they are populated per category on expand rather than
decoding 85 .dds files the moment the view is built.
"""

from __future__ import annotations

from PyQt5.QtCore import QSize, Qt, pyqtSignal
from PyQt5.QtGui import QIcon, QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QButtonGroup, QHBoxLayout, QStackedWidget, QToolButton, QTreeWidgetItem

from descape import asset_source, terrain_catalog, terrain_palette
from descape.swatch_grid import SwatchGridView
from descape.value_picker import PickerItem, ValuePickerView

HIDDEN_LABEL = "Show unused / moddable terrains"
SWATCH_PX = asset_source.TERRAIN_THUMBNAIL_PX
ROW_ICON_PX = 32

# Stated in the picker page rather than only in the code: 131 terrains share 85
# textures, and the thing that really distinguishes FOREST_OAK from
# FOREST_PINE is the trees scattered on the tile, which DEscape does not
# draw in the map render either. So a bare swatch IS what painting this
# terrain will look like here -- a pre-existing limit, not one this picker
# introduces.
SHARED_TEXTURE_NOTE = (
    "Forest and snow variants share a ground texture; the trees that "
    "distinguish them aren't drawn yet."
)


def swatch_pixmap(terrain_id: int, px: int = SWATCH_PX) -> QPixmap:
    """terrain_id's real texture at px square, or a flat fill of its palette
    colour when no install is configured. Never None: a blank row would read
    as a broken swatch rather than as "no install", and
    color_for_terrain_id already falls back for exactly this case."""
    array = asset_source.get_terrain_thumbnail(terrain_id, px)
    if array is None:
        pixmap = QPixmap(px, px)
        pixmap.fill(_flat_color(terrain_id))
        return pixmap
    import numpy as np

    array = np.ascontiguousarray(array)
    height, width = array.shape[:2]
    image = QImage(array.data, width, height, 3 * width, QImage.Format_RGB888)
    # fromImage() copies, so `array` going out of scope here can't leave the
    # pixmap holding a dangling buffer.
    return QPixmap.fromImage(image)


def _flat_color(terrain_id: int):
    from PyQt5.QtGui import QColor

    return QColor(*terrain_palette.color_for_terrain_id(terrain_id))


def items_for(entries=None) -> tuple[PickerItem, ...]:
    """terrain_catalog rows reduced to value_picker's domain-free shape. The
    search text carries the RAW enum name as well as the display label, so
    filtering on "non_navigable" works as well as on "non navigable"."""
    entries = terrain_catalog.terrains() if entries is None else entries
    return tuple(
        PickerItem(
            label=terrain_catalog.display_name(entry.name),
            value=entry.id,
            group=entry.category,
            hidden=entry.hidden,
            search_text=f"{entry.name.lower()} {terrain_catalog.display_name(entry.name).lower()} {entry.id}",
        )
        for entry in entries
    )


def _category_of(terrain_id: int | None) -> str:
    """The category to start expanded, or "" for none matched -- which
    ValuePickerView reads as "expand everything", so a caller passing an
    unknown id gets the old all-expanded behaviour rather than an empty
    tree."""
    if terrain_id is None:
        return ""
    for entry in terrain_catalog.terrains():
        if entry.id == terrain_id:
            return entry.category
    return ""


class TerrainPickerView(ValuePickerView):
    """Pick a terrain by sight. Grouped by category, filterable, with the
    junk terrains behind the "show unused" box.

    A plain always-visible widget, not a dialog: TerrainPanel embeds one as
    the Terrain-mode sidebar page. Being long-lived is what refresh_swatches()
    below exists for.

    Two presentations behind the List/Grid buttons beside the filter (GH
    #161): the tree, and `self.grid`, a SwatchGridView. The tree stays the one
    source of truth for selection and filtering, so select(), current_value()
    and current_changed mean the same in both modes; the grid mirrors it.
    """

    display_mode_changed = pyqtSignal(str)

    def __init__(self, current_id: int | None = None, parent=None, *, display_mode: str = "list", swatch_px: int = 48):
        self._mode = "list"
        self._mirroring = False
        # default_group is load-bearing, not a nicety: ValuePickerView
        # expands EVERY group when it is empty, and an all-expanded tree
        # decodes all 85 distinct textures at once -- measured at 2.8s cold,
        # which is the stall the per-category population exists to avoid.
        # Opening on the current terrain's own category collapses the other
        # nine and lands the user where they already are.
        super().__init__(
            items_for(),
            default_group=_category_of(current_id),
            hidden_label=HIDDEN_LABEL,
            parent=parent,
        )
        self._build_grid(swatch_px)
        self.tree.setIconSize(QSize(ROW_ICON_PX, ROW_ICON_PX))
        # ValuePickerView turns uniform row heights ON, which is right for
        # its text-only consumers and wrong here: Qt fixes that height from
        # the first row, which exists before any icon does, so every swatch
        # renders squeezed into a 19px text row (measured) instead of its
        # own 32px square. Off costs nothing at 131 rows.
        self.tree.setUniformRowHeights(False)
        # Per category on expand, not all at once: 85 distinct .dds decodes
        # up front is a visible stall on open, and most categories are
        # collapsed. _apply_filter expands a group programmatically when a
        # search reveals it, and setExpanded() emits itemExpanded, so a
        # filter-revealed row is covered by this same connection.
        self.tree.itemExpanded.connect(self._populate_icons)
        for i in range(self.tree.topLevelItemCount()):
            group = self.tree.topLevelItem(i)
            if group.isExpanded():
                self._populate_icons(group)

        if current_id is not None:
            self.select(current_id)
        if display_mode != "list":
            self.set_display_mode(display_mode)

    # -- the grid -------------------------------------------------------------

    def _build_grid(self, swatch_px: int) -> None:
        """The stack (tree, grid) in the tree's place, and the List/Grid buttons beside the filter."""
        # Looked up at call time, so a test's monkeypatched swatch_pixmap reaches the grid too.
        hook = lambda value: swatch_pixmap(value)  # noqa: E731, PLW0108
        self.grid = SwatchGridView(items_for(), swatch=hook, swatch_px=swatch_px)
        self.stack = QStackedWidget()
        self._tree_row.removeWidget(self.tree)
        self.stack.addWidget(self.tree)
        self.stack.addWidget(self.grid)
        self._tree_row.insertWidget(0, self.stack, stretch=1)

        layout = self.layout()
        filter_row = QHBoxLayout()
        layout.removeWidget(self.filter_edit)
        filter_row.addWidget(self.filter_edit, stretch=1)
        self.mode_buttons: dict[str, QToolButton] = {}
        group = QButtonGroup(self)
        group.setExclusive(True)
        for mode, label, tip in (("list", "List", "Show terrains as a list"),
                                 ("grid", "Grid", "Show terrains as a grid of swatches")):
            button = QToolButton()
            button.setText(label)
            button.setToolTip(tip)
            button.setCheckable(True)
            button.setFocusPolicy(Qt.NoFocus)
            button.clicked.connect(lambda _checked, m=mode: self.set_display_mode(m))
            group.addButton(button)
            filter_row.addWidget(button)
            self.mode_buttons[mode] = button
        self.mode_buttons["list"].setChecked(True)
        layout.insertLayout(0, filter_row)

        self.grid.picked.connect(self._on_grid_picked)
        self.grid.activated.connect(self.activated.emit)
        self.tree.currentItemChanged.connect(lambda *_: self._mirror_to_grid())
        # The base __init__ filtered before the grid existed.
        self._apply_filter(self.filter_edit.text())
        self._mirror_to_grid()

    def display_mode(self) -> str:
        return self._mode

    def set_display_mode(self, mode: str) -> None:
        """Flag, then page, then the new page's own upkeep: with the flag
        flipped last, the expand Qt performs on show would hit
        _populate_icons' grid-mode return and leave rows iconless."""
        if mode not in ("list", "grid"):
            raise ValueError(f"display mode must be 'list' or 'grid', got {mode!r}")
        if mode == self._mode:
            return
        self._mode = mode
        self.stack.setCurrentWidget(self.grid if mode == "grid" else self.tree)
        self.mode_buttons[mode].setChecked(True)
        if mode == "list":
            for i in range(self.tree.topLevelItemCount()):
                group = self.tree.topLevelItem(i)
                if group.isExpanded():
                    self._populate_icons(group)
            if self.tree.currentItem() is not None:
                self.tree.scrollToItem(self.tree.currentItem())
        else:
            self.grid.scroll_to_value(self.grid.current_value())
        self.display_mode_changed.emit(mode)

    def set_grid_swatch_px(self, px: int) -> None:
        self.grid.set_swatch_px(px)

    def _on_grid_picked(self, value: int) -> None:
        self._mirroring = True
        try:
            self.select(value)
        finally:
            self._mirroring = False

    def _mirror_to_grid(self) -> None:
        # A heading or a filtered-away row leaves the grid's frame on the armed terrain (TerrainPanel's sticky rule).
        if self._mirroring or not hasattr(self, "grid"):
            return
        value = self.current_value()
        if value is not None:
            self.grid.set_current_value(value)

    def _apply_filter(self, text: str) -> None:
        super()._apply_filter(text)
        if not hasattr(self, "grid"):
            return
        shown = set()
        for i in range(self.tree.topLevelItemCount()):
            group = self.tree.topLevelItem(i)
            if group.isHidden():
                continue
            for c in range(group.childCount()):
                child = group.child(c)
                if not child.isHidden():
                    shown.add(child.data(0, Qt.UserRole))
        self.grid.set_visible_values(shown)

    def refresh_swatches(self) -> None:
        """Drop every row icon and redecode the expanded groups.

        Needed only because this view outlives an install-path change:
        _populate_icons skips any row that already has an icon, so without
        the clear the tree would keep showing the flat fallback colours (or
        the previous install's textures) forever.
        """
        for i in range(self.tree.topLevelItemCount()):
            group = self.tree.topLevelItem(i)
            for c in range(group.childCount()):
                group.child(c).setIcon(0, QIcon())
        for i in range(self.tree.topLevelItemCount()):
            group = self.tree.topLevelItem(i)
            if group.isExpanded():
                self._populate_icons(group)
        self.grid.refresh_swatches()

    def _populate_icons(self, group: QTreeWidgetItem) -> None:
        # In Grid the tree is hidden, and select() still expands groups: decoding them
        # here would stall under a wait cursor for rows nobody sees. List mode catches up.
        if self._mode == "grid":
            return
        pending = [
            child
            for c in range(group.childCount())
            for child in [group.child(c)]
            if child.icon(0).isNull() and isinstance(child.data(0, Qt.UserRole), int)
        ]
        if not pending:
            return
        # A .dds decode measures ~28ms here, so expanding Forest (28 rows,
        # 12 of them sharing g_for.dds) is a visible fraction of a second.
        # Same wait-cursor convention ViewerWindow.on_fill uses, for the same
        # reason: cheap, and the alternative is a window that looks hung.
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            for child in pending:
                # The 64 px swatch, drawn at ROW_ICON_PX by the view: list and grid share one decode.
                child.setIcon(0, QIcon(swatch_pixmap(child.data(0, Qt.UserRole))))
        finally:
            QApplication.restoreOverrideCursor()
