"""The visual terrain picker: a category-grouped tree of terrain swatches,
living in the Terrain-mode sidebar page (descape/terrain_panel.py).

A thin adapter over descape/value_picker.py, the same shape
constant_picker.py's CatalogBrowseDialog is. (The plan for this called for a
hand-written sibling instead, because CatalogBrowseDialog was hard-wired to
object_catalog/unit_sprites at the time -- the value_picker extraction has
since lifted exactly that coupling out, so there is nothing left to copy.)

What this module adds on top is the one piece of domain knowledge
value_picker.py has none of: a terrain swatch, as both the side preview and
a per-row icon. The icons are what make this a browser rather than a second
list of names, so they are populated per category on expand rather than
decoding 85 .dds files the moment the view is built.
"""

from __future__ import annotations

from PyQt5.QtCore import QSize, Qt
from PyQt5.QtGui import QIcon, QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QTreeWidgetItem

from descape import asset_source, terrain_catalog, terrain_palette
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


def _terrain_preview(item: PickerItem) -> QPixmap:
    return swatch_pixmap(item.value)


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
    """

    def __init__(self, current_id: int | None = None, parent=None):
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
            preview=_terrain_preview,
            parent=parent,
        )
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

    def refresh_swatches(self) -> None:
        """Drop every row icon and redecode the expanded groups.

        Needed only because this view outlives an install-path change:
        _populate_icons skips any row that already has an icon, so without
        the clear the tree would keep showing the flat fallback colours (or
        the previous install's textures) forever. The side preview is
        rebuilt too, since it is a separate pixmap that only refreshes when
        the current row changes.
        """
        for i in range(self.tree.topLevelItemCount()):
            group = self.tree.topLevelItem(i)
            for c in range(group.childCount()):
                group.child(c).setIcon(0, QIcon())
        for i in range(self.tree.topLevelItemCount()):
            group = self.tree.topLevelItem(i)
            if group.isExpanded():
                self._populate_icons(group)
        self._update_preview()

    def _populate_icons(self, group: QTreeWidgetItem) -> None:
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
                child.setIcon(0, QIcon(swatch_pixmap(child.data(0, Qt.UserRole), ROW_ICON_PX)))
        finally:
            QApplication.restoreOverrideCursor()
