"""CatalogBrowserView: the Units panel's object picker (GH #68, GH #6), and
Browse's (GH #273).

Like the in-game editor's object menus: a 2x2 block of category buttons
(Units, Buildings, Heroes, Other), a flat list of the active category, a
class combo that narrows it, a name-or-id filter and three display modes
(icon and name rows, text rows, an icon-only grid) in one dropdown. ValuePickerView stays the
picker for every other consumer; this one reads object_catalog.objects()
directly, since it needs each entry's class, plus place_random's pseudo rows
(GH #164). Browse hands it `entries` instead, one field's catalog with no
random rows, and a `mode_store` of its own.

The API the panel and its tests use is ValuePickerView's: `activated(int)`,
`current_changed(object)`, select(), current_value(), `filter_edit`,
`show_hidden_checkbox`, pin(); plus `picked(int)` (a user's click on a row,
the current one included, or a keyboard move in the list: GH #357), `list`,
`category_buttons`, `class_combo`, `display_combo` and refresh_thumbnails().

With a `favorites` store (the sidebar's, GH #195; Browse passes none) it also
has a Favorites toggle that lists only the favorites in the user's order, a
right-click menu that adds or removes one (or, with settings'
units_catalog_right_click "toggle", a right-click that does so at once), a star
on every favorite (settings.favorite_star_style: outlined gold or black and
white) and drag reorder in the favorites view. Ids the dataset lacks stay in the store
and show greyed as "Unknown object (id)": selectable, never picked. The list
widget, its delegate and the row helpers are shared with favorites_dialog.

Rows of the active category stay in the list and are hidden by the filter,
the class combo and the editor-hidden box, so select() can make a filtered-out
row current the way ValuePickerView's tree does (UnitsPanel.pick_object relies
on that). Thumbnails come from the panel's shared catalog_thumbs.ThumbLoader,
requested for the rows in the viewport only.
"""

from __future__ import annotations

import dataclasses
import math
from collections import Counter
from collections.abc import Callable, Sequence

from AoE2ScenarioParser.datasets.trigger_lists import ObjectClass
from PyQt5.QtCore import QPoint, QPointF, QRect, QSize, Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QBrush, QColor, QFont, QIcon, QPainter, QPalette, QPen, QPixmap, QPolygonF
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QGridLayout,
    QHBoxLayout,
    QLineEdit,
    QListView,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QPushButton,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from descape import asset_source, catalog_thumbs, item_activation, object_catalog, place_random, settings, themes
from descape.swatch_grid import FRAME_PX
from descape.terrain_palette import PORTRAIT_COLORS

HIDDEN_LABEL = "Show editor-hidden objects"
# (object_catalog category, button label), in the in-game editor's order, two to a row.
CATEGORIES = (("Units", "Units"), ("Buildings", "Buildings"), ("Heroes", "Heroes"), ("Others", "Other"))
# Display mode -> (dropdown label, tooltip).
_MODES = {
    "icons": ("Icons", "Icon and name rows"),
    "names": ("Names", "Name rows only"),
    "grid": ("Grid", "Icons only, name and Unit ID on hover"),
}
_NO_INSTALL_TIP = "Needs the game install (Settings), which supplies the portraits and sprites"
_NO_INSTALL_COMBO_TIP = "Icons and Grid need the game install (Settings), which supplies the portraits and sprites"
_LIST_ICON_PX = 32
_GRID_ICON_PX = 48
_GRID_CELL_PAD = 8
_ALL_CLASSES = "All classes"
# The class combo's width in characters, so it, Favorites and the display dropdown fit MIN_USEFUL_WIDTH.
_CLASS_COMBO_CHARS = 8
_VALUE_ROLE = Qt.UserRole
# GH #195: set on a favorite's row, read by the delegate to draw its star.
FAVORITE_ROLE = Qt.UserRole + 1
# The favorites view's category key: never a real object_catalog category.
_FAVORITES = "\x00favorites"
FAVORITES_TIP = "Show only your favorites. Right-click an object to add or remove one."
_STAR_GOLD = QColor(232, 176, 20)
# The gold star's edge: gold alone is 1.02:1 on the Wood themes and under 2:1 on the light ones.
_STAR_OUTLINE = QColor(90, 60, 0)
_STAR_OUTLINE_PX = 1.3
# A list row's star: its outer radius and the room it keeps at the row's right end.
_STAR_RADIUS = 6
_STAR_ROOM = 2 * _STAR_RADIUS + 8


def class_label(class_id: int | None) -> str:
    """ObjectClass's name in words ("Cavalry Archer"), or "Class N" for one it lacks."""
    try:
        return ObjectClass(class_id).name.replace("_", " ").title()
    except ValueError:
        return f"Class {class_id}"


def _initials(name: str) -> str:
    words = [word for word in name.replace("(", " ").split() if word[:1].isalnum()]
    return "".join(word[0] for word in words[:2]).upper() or "?"


def _glyph(name: str, px: int) -> QIcon:
    """A grid cell's stand-in when a row has no thumbnail: never a blank cell."""
    pixmap = QPixmap(px, px)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    painter.setPen(QColor(128, 128, 128))
    painter.setBrush(QColor(128, 128, 128, 48))
    painter.drawRoundedRect(QRect(1, 1, px - 2, px - 2), 6, 6)
    font = QFont(painter.font())
    font.setPixelSize(max(8, px // 3))
    font.setBold(True)
    painter.setFont(font)
    painter.drawText(QRect(0, 0, px, px), Qt.AlignCenter, _initials(name))
    painter.end()
    return QIcon(pixmap)


def _random_badge(icon: QIcon, px: int) -> QIcon:
    """A random pick's preview thumbnail with a "?" in its corner, so it never
    reads as the real object beside it (GH #164)."""
    pixmap = QPixmap(px, px)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    if icon is not None:
        source = icon.pixmap(px, px)
        painter.drawPixmap((px - source.width()) // 2, (px - source.height()) // 2, source)
    badge = max(10, px * 2 // 5)
    corner = QRect(px - badge, px - badge, badge, badge)
    painter.setPen(Qt.NoPen)
    painter.setBrush(QColor(40, 40, 40, 220))
    painter.drawEllipse(corner)
    font = QFont(painter.font())
    font.setPixelSize(max(8, badge * 3 // 4))
    font.setBold(True)
    painter.setFont(font)
    painter.setPen(QColor(255, 255, 255))
    painter.drawText(corner, Qt.AlignCenter, "?")
    painter.end()
    return QIcon(pixmap)


def thumb_const(value: int) -> int:
    """The const whose thumbnail a row shows: a random pick shows its preview const's."""
    return place_random.preview_const(value) if place_random.is_random(value) else value


def unknown_label(value: int) -> str:
    """An id the active dataset does not list (GH #195): kept, never placed."""
    return f"Unknown object ({value})"


def owner_renames(old: int | None, civ: int | None) -> bool:
    """Whether an Owner switch from civ `old` to `civ` renames any row (GH #372):
    the two civs' own names differ, in which ids or in the names themselves."""
    named = object_catalog.civ_named_ids(civ)
    return object_catalog.civ_named_ids(old) != named or any(
        object_catalog.civ_name(i, old) != object_catalog.civ_name(i, civ) for i in named
    )


def owner_entries(entries, civ: int | None):
    """Each entry (None passes through) named as civ's player sees it (an Athenian Archer reads Bowman)."""
    named = object_catalog.civ_named_ids(civ)
    for entry in entries:
        if entry is None or entry.id not in named:
            yield entry
            continue
        name = object_catalog.owner_name(entry.id, civ, entry.name)
        # search_text stays the standard name's, so "archer" still finds the Bowman.
        yield entry if name == entry.name else dataclasses.replace(entry, name=name)


def row_tooltip(entry, favorite: bool) -> str:
    text = f"{entry.name} ({entry.id})" if entry.id >= 0 else entry.name
    return f"{text} (favorite)" if favorite else text


def placeholder_icon(px: int) -> QIcon:
    pixmap = QPixmap(px, px)
    pixmap.fill(Qt.transparent)
    return QIcon(pixmap)


def mode_combo() -> QComboBox:
    """A display dropdown with the three modes (the catalog's and the Favorite Objects window's);
    sync_mode_combo() sets its state and tooltips."""
    combo = QComboBox()
    combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
    for mode, (label, _tip) in _MODES.items():
        combo.addItem(label, mode)
    return combo


def fill_rows(view: QListWidget, rows, mode: str, favorites) -> dict[int, QListWidgetItem]:
    """Refills `view` with a row per (id, CatalogEntry or None) in `rows`, in `mode`, and returns
    them by id. None is an id the active dataset lacks, greyed. Signals and the current row are the caller's."""
    grid = mode == "grid"
    px = icon_px(mode)
    # A blank icon of the mode's size: a row has its full size before its thumbnail lands,
    # and a uniform-size grid does not size every cell from an icon-less first one.
    placeholder = None if px is None else placeholder_icon(px)
    unknown_text = QBrush(view.palette().color(QPalette.Disabled, QPalette.Text))
    view.clear()
    items: dict[int, QListWidgetItem] = {}
    for value, entry in rows:
        if entry is None:
            item = QListWidgetItem("" if grid else unknown_label(value))
            item.setToolTip(f"{unknown_label(value)}: not in this game's data, so it cannot be placed")
            item.setForeground(unknown_text)
            if px is not None:
                item.setIcon(_glyph("?", px) if grid else placeholder)
        else:
            item = QListWidgetItem("" if grid else entry.name)
            item.setToolTip(row_tooltip(entry, value in favorites))
            if placeholder is not None:
                item.setIcon(placeholder)
        item.setData(_VALUE_ROLE, value)
        item.setData(FAVORITE_ROLE, value in favorites)
        view.addItem(item)
        items[value] = item
    return items


def configure_list(view: QListWidget, mode: str) -> None:
    """A catalog list's layout for one display mode (the catalog's and the Favorite Objects window's)."""
    if mode == "grid":
        view.setViewMode(QListView.IconMode)
        view.setIconSize(QSize(_GRID_ICON_PX, _GRID_ICON_PX))
        view.setGridSize(QSize(_GRID_ICON_PX + _GRID_CELL_PAD, _GRID_ICON_PX + _GRID_CELL_PAD))
        view.setResizeMode(QListView.Adjust)
        view.setMovement(QListView.Static)
        view.setWrapping(True)
        view.setUniformItemSizes(True)
    else:
        view.setViewMode(QListView.ListMode)
        view.setGridSize(QSize())
        view.setWrapping(False)
        view.setMovement(QListView.Static)
        view.setResizeMode(QListView.Fixed)
        view.setIconSize(QSize(_LIST_ICON_PX, _LIST_ICON_PX) if mode == "icons" else QSize())
        # Uniform sizes take every row's width from one row, clipping a longer name instead of scrolling to it.
        view.setUniformItemSizes(False)


def icon_px(mode: str) -> int | None:
    return {"icons": _LIST_ICON_PX, "grid": _GRID_ICON_PX}.get(mode)


def set_drag_reorder(view: QListWidget, on: bool) -> None:
    """Drag to reorder (GH #195). After configure_list(): its Static movement turns drag off."""
    view.setDragEnabled(on)
    view.setAcceptDrops(on)
    # setMovement(Static) turns the viewport's off directly; the view's own setter skips an unchanged value.
    view.viewport().setAcceptDrops(on)
    view.setDropIndicatorShown(on)
    view.setDragDropMode(QAbstractItemView.InternalMove if on else QAbstractItemView.NoDragDrop)


def visible_values(view: QListWidget) -> list[int]:
    """The rows on screen, top to bottom, found by indexAt from the
    viewport's top-left and walked forward until a row starts below it."""
    viewport = view.viewport().rect()
    first = view.indexAt(QPoint(2, 2))
    row = first.row() if first.isValid() else 0
    values = []
    for index in range(max(row - 1, 0), view.count()):
        item = view.item(index)
        if item.isHidden():
            continue
        rect = view.visualItemRect(item)
        if rect.top() > viewport.bottom():
            break
        if rect.intersects(viewport):
            values.append(item.data(_VALUE_ROLE))
    return values


def thumb_fits(const: int, px: int, civ: int | None, want_px: int | None, owner_civ: int | None) -> bool:
    """Whether a ready thumbnail is this list's: its size, and its Owner's portrait civ (GH #372)."""
    return px == want_px and civ == object_catalog.portrait_civ_key(const, owner_civ)


def row_icon(loader, value: int, icon, px: int, civ: int | None, team_rgb, name: str, grid: bool) -> QIcon:
    """A row's icon from a ready thumbnail: the team colour filled in, a random pick's badge, and
    a stand-in when there is no thumbnail (the placeholder, or Grid's initials)."""
    const = thumb_const(value)
    source = loader.source(const, px, civ)
    icon = catalog_thumbs.team_filled(icon, source, team_rgb, px, loader.art(const, px, civ))
    if place_random.is_random(value):
        icon = _random_badge(icon, px)
    if icon is None:
        # The placeholder, not nothing: the row may still show another owner's portrait (GH #372).
        icon = _glyph(name, px) if grid else placeholder_icon(px)
    return icon


def mono_star_colour(palette: QPalette) -> QColor:
    """Black and white's list-row star: pure black or white, whichever contrasts more with the rows
    (a theme's own text colour can be a dim cream, as on Dark Wood)."""
    base = palette.color(QPalette.Base).name()
    white, black = "#ffffff", "#000000"
    return QColor(white if themes.contrast_ratio(white, base) >= themes.contrast_ratio(black, base) else black)


def _star(center: QPointF, radius: float) -> QPolygonF:
    points = []
    for index in range(10):
        r = radius if index % 2 == 0 else radius * 0.45
        angle = math.pi / 2 + index * math.pi / 5
        points.append(QPointF(center.x() + r * math.cos(angle), center.y() - r * math.sin(angle)))
    return QPolygonF(points)


class CatalogList(QListWidget):
    """Scroll and resize both change which rows are on screen. A right press
    changes no row (the favorites menu is the right button's), and a drag
    within the list reports the new order instead of moving items itself."""

    viewport_changed = pyqtSignal()
    # A key press moved the current row: GH #357's keyboard pick, apart from a programmatic setCurrentItem().
    key_moved = pyqtSignal()
    # GH #195: every row's value in its new order after a drop, emitted on the next event loop turn.
    reordered = pyqtSignal(list)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self.viewport_changed.emit()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.RightButton:
            event.accept()
            return
        super().mousePressEvent(event)

    # Qt's own handlers first in enter and move: the dragging state, the drop indicator and autoscroll.
    def dragEnterEvent(self, event) -> None:
        super().dragEnterEvent(event)
        if event.source() is self:
            event.accept()

    def dragMoveEvent(self, event) -> None:
        super().dragMoveEvent(event)
        if event.source() is self:
            event.setDropAction(Qt.MoveAction)
            event.accept()

    def dropEvent(self, event) -> None:
        if event.source() is not self:
            super().dropEvent(event)
            return
        order = self.drop_order(event.pos())
        # Copy: Qt's startDrag removes the dragged row after a MoveAction drop.
        event.setDropAction(Qt.CopyAction)
        event.accept()
        # Out of the dragging state, as Qt's dropEvent leaves it, so the indicator goes at once.
        self.setState(QAbstractItemView.NoState)
        self.viewport().update()
        if order is not None:
            QTimer.singleShot(0, lambda: self.reordered.emit(order))

    def drop_order(self, pos: QPoint) -> list[int] | None:
        """Every row's value with the current row moved to `pos` (after the row there
        when `pos` is in its second half), or None when nothing moves."""
        moved = self.currentItem()
        if moved is None:
            return None
        values = [self.item(index).data(_VALUE_ROLE) for index in range(self.count())]
        value = moved.data(_VALUE_ROLE)
        target = self._drop_target(pos)
        if target is None:
            slot = len(values)
        else:
            slot = self.row(target)
            rect = self.visualItemRect(target)
            second_half = pos.x() > rect.center().x() if self.viewMode() == QListView.IconMode else pos.y() > rect.center().y()
            slot += int(second_half)
        if slot > values.index(value):
            slot -= 1
        values.remove(value)
        values.insert(slot, value)
        current = [self.item(index).data(_VALUE_ROLE) for index in range(self.count())]
        return None if values == current else values

    def _drop_target(self, pos: QPoint) -> QListWidgetItem | None:
        """The row under `pos`, else the nearest one (Grid's gaps between cells and the
        strip after a line), or None below the last row: the end of the list."""
        target = self.itemAt(pos)
        if target is not None:
            return target
        shown = [self.item(index) for index in range(self.count()) if not self.item(index).isHidden()]
        if not shown or pos.y() > self.visualItemRect(shown[-1]).bottom():
            return None

        def distance(item) -> int:
            rect = self.visualItemRect(item)
            dx = max(rect.left() - pos.x(), 0, pos.x() - rect.right())
            dy = max(rect.top() - pos.y(), 0, pos.y() - rect.bottom())
            return dx * dx + dy * dy

        return min(shown, key=distance)

    def keyPressEvent(self, event) -> None:
        before = self.currentItem()
        super().keyPressEvent(event)
        if self.currentItem() is not before:
            self.key_moved.emit()


class CatalogDelegate(QStyledItemDelegate):
    """The Grid's selected cell: the same Highlight frame as swatch_grid's, not the style's faint tint.
    A favorite (FAVORITE_ROLE) also gets a star (GH #195): at its row's right end in the list
    modes, gold with a dark outline or, in the "mono" style, black or white by the row colour
    (the highlighted text colour on a selected row); on a dark disc in its cell's top-right
    corner in Grid, gold or white."""

    def paint(self, painter, option, index) -> None:
        view = self.parent()
        grid = view.viewMode() == QListView.IconMode
        selected = bool(option.state & QStyle.State_Selected)
        if not grid or not selected:
            super().paint(painter, option, index)
        else:
            untinted = QStyleOptionViewItem(option)
            untinted.state &= ~QStyle.State_Selected
            super().paint(painter, untinted, index)
            painter.save()
            pen = QPen(view.palette().color(QPalette.Highlight))
            pen.setWidth(FRAME_PX)
            pen.setJoinStyle(Qt.MiterJoin)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            half = FRAME_PX // 2
            painter.drawRect(option.rect.adjusted(half, half, -half, -half))
            painter.restore()
        if index.data(FAVORITE_ROLE):
            mono = settings.get_favorite_star_style() == "mono"
            self._paint_star(painter, option.rect, grid, selected, view.palette(), mono)

    @staticmethod
    def _paint_star(painter, rect: QRect, grid: bool, selected: bool, palette: QPalette, mono: bool = False) -> None:
        painter.save()
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(Qt.NoPen)
        if grid:
            disc = 2 * _STAR_RADIUS + 4
            corner = QRect(rect.right() - disc - 1, rect.top() + 2, disc, disc)
            painter.setBrush(QColor(40, 40, 40, 220))
            painter.drawEllipse(corner)
            painter.setBrush(QColor(255, 255, 255) if mono else _STAR_GOLD)
            painter.drawPolygon(_star(QPointF(corner.center()) + QPointF(0.5, 0.5), _STAR_RADIUS - 1))
        else:
            if selected:
                painter.setBrush(palette.color(QPalette.HighlightedText))
            elif mono:
                painter.setBrush(mono_star_colour(palette))
            else:
                pen = QPen(_STAR_OUTLINE, _STAR_OUTLINE_PX)
                pen.setJoinStyle(Qt.MiterJoin)
                painter.setPen(pen)
                painter.setBrush(_STAR_GOLD)
            center = QPointF(rect.right() - _STAR_ROOM / 2, rect.center().y() + 0.5)
            painter.drawPolygon(_star(center, _STAR_RADIUS))
        painter.restore()

    def sizeHint(self, option, index) -> QSize:
        hint = super().sizeHint(option, index)
        if index.data(FAVORITE_ROLE) and self.parent().viewMode() != QListView.IconMode:
            # Room for the star, so a long name scrolls rather than running under it.
            hint.setWidth(hint.width() + _STAR_ROOM)
        return hint


class CatalogBrowserView(QWidget):
    activated = pyqtSignal(int)  # itemActivated: double-click and Enter, in every mode
    current_changed = pyqtSignal(object)  # int | None
    picked = pyqtSignal(int)  # a click on a visible row, or a key that moved the current row
    favorites_changed = pyqtSignal()  # GH #195: the stored favorites (or their order) changed
    favorites_view_left = pyqtSignal()  # the Favorites button or a category click turned the view off

    def __init__(
        self,
        loader: catalog_thumbs.ThumbLoader | None = None,
        parent: QWidget | None = None,
        *,
        entries: Sequence[object_catalog.CatalogEntry] | None = None,
        default_category: str = "",
        mode_store: tuple[Callable[[], str], Callable[[str], None]] | None = None,
        favorites: tuple[Callable[[], list[int]], Callable[[list[int]], None]] | None = None,
    ):
        """`entries`: those rows and no random ones, reload_entries() being the
        sidebar's alone. `default_category` opens if it has rows, else the first
        category that does. `mode_store`: (getter, setter) for the display mode,
        by default the sidebar's own setting. `favorites`: (getter, setter) for the
        favorite ids; None leaves out the button, the menu and the star."""
        super().__init__(parent)
        self._loader = loader if loader is not None else catalog_thumbs.ThumbLoader(self)
        self._loader.ready.connect(self._on_thumb_ready)
        if entries is None:
            self._entries, self._by_id = self._load_entries()
        else:
            self._entries = tuple(entries)
            self._by_id = {entry.id: entry for entry in self._entries}
        # The rows as loaded; _entries and _by_id carry set_civ()'s owner names over them.
        self._base_entries, self._base_by_id = self._entries, self._by_id
        self._items: dict[int, QListWidgetItem] = {}
        self._category = self._opening_category(default_category)
        # The real category the favorites view returns to.
        self._last_category = self._category
        get_mode, self._save_mode = mode_store or (settings.get_units_catalog_mode, settings.set_units_catalog_mode)
        self._mode = get_mode()
        self._favorites_store = favorites
        # pin()'s value: listed whatever the hidden box says.
        self._pinned: int | None = None
        # Behind a portrait's player-colour region: the Owner combo's, pushed by set_team_color().
        self._team_rgb: tuple[int, int, int] = PORTRAIT_COLORS[0]
        # The Owner combo's portrait civ (GH #372), pushed by set_civ(); None shows GAIA's.
        self._civ: int | None = None
        # GH #32: consts set_excluded() hides, and the row values that hides.
        self._excluded: frozenset[int] = frozenset()
        self._excluded_rows: frozenset[int] = frozenset()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.category_buttons: dict[str, QPushButton] = {}
        self._category_group = QButtonGroup(self)
        self._category_group.setExclusive(True)
        # Its own widget so _limit_categories() can hide the block whole.
        self.category_block = QWidget()
        grid = QGridLayout(self.category_block)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(2)
        for index, (category, label) in enumerate(CATEGORIES):
            button = QPushButton(label)
            button.setCheckable(True)
            button.setFocusPolicy(Qt.NoFocus)
            button.setProperty("label", label)
            button.clicked.connect(lambda _checked, c=category: self._on_category_clicked(c))
            self._category_group.addButton(button)
            self.category_buttons[category] = button
            grid.addWidget(button, index // 2, index % 2)
        self.category_buttons[self._category].setChecked(True)
        layout.addWidget(self.category_block)
        if entries is not None:
            self._limit_categories()

        class_row = QHBoxLayout()
        class_row.setContentsMargins(0, 0, 0, 0)
        self.class_combo = QComboBox()
        self.class_combo.setFocusPolicy(Qt.ClickFocus)
        # Sized by a character count, not its longest class; the popup shows each in full.
        self.class_combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.class_combo.setMinimumContentsLength(_CLASS_COMBO_CHARS)
        self.class_combo.currentIndexChanged.connect(lambda _index: self._apply_filter())
        class_row.addWidget(self.class_combo, stretch=1)
        self.favorites_button: QToolButton | None = None
        if favorites is not None:
            # A tool button: a push button's style minimum (about 80 px) would set the pane's floor.
            self.favorites_button = QToolButton()
            self.favorites_button.setText("Favorites")
            self.favorites_button.setCheckable(True)
            self.favorites_button.setFocusPolicy(Qt.NoFocus)
            self.favorites_button.setToolTip(FAVORITES_TIP)
            self.favorites_button.clicked.connect(self._on_favorites_clicked)
            class_row.addWidget(self.favorites_button)
        # One dropdown for the three modes (GH #195 freed their buttons' room for Favorites).
        self.display_combo = mode_combo()
        self.display_combo.setFocusPolicy(Qt.NoFocus)
        # activated, not currentIndexChanged: _sync_mode_combo()'s own index change must never save.
        self.display_combo.activated.connect(lambda index: self._choose_mode(self.display_combo.itemData(index)))
        class_row.addWidget(self.display_combo)
        layout.addLayout(class_row)

        filter_row = QHBoxLayout()
        filter_row.setContentsMargins(0, 0, 0, 0)
        self.filter_edit = QLineEdit()
        self.filter_edit.setPlaceholderText("Filter…")
        self.filter_edit.setClearButtonEnabled(True)
        self.filter_edit.textChanged.connect(lambda _text: self._on_filter_changed())
        filter_row.addWidget(self.filter_edit, stretch=1)
        self.show_hidden_checkbox = QCheckBox(HIDDEN_LABEL)
        self.show_hidden_checkbox.toggled.connect(lambda _on: self._on_hidden_toggled())
        filter_row.addWidget(self.show_hidden_checkbox)
        layout.addLayout(filter_row)

        self.list = CatalogList()
        # ClickFocus: MapView's arrow-key nudge needs the map to keep focus (units_panel.py).
        self.list.setFocusPolicy(Qt.ClickFocus)
        self.list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.list.setTextElideMode(Qt.ElideNone)
        self.list.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.list.setItemDelegate(CatalogDelegate(self.list))
        self.list.itemActivated.connect(self._on_activated)
        item_activation.install(self.list)
        self.list.currentItemChanged.connect(lambda *_: self.current_changed.emit(self.current_value()))
        # itemClicked, since a click on the current row changes nothing currentItemChanged reports.
        self.list.itemClicked.connect(self._on_clicked)
        self.list.key_moved.connect(self._on_key_moved)
        self.list.verticalScrollBar().valueChanged.connect(lambda _value: self._request_viewport())
        self.list.horizontalScrollBar().valueChanged.connect(lambda _value: self._request_viewport())
        self.list.viewport_changed.connect(self._request_viewport)
        self.list.reordered.connect(self.set_favorites)
        if favorites is not None:
            self.list.setContextMenuPolicy(Qt.CustomContextMenu)
            self.list.customContextMenuRequested.connect(self._show_favorite_menu)
        layout.addWidget(self.list, stretch=1)

        self._populate()
        self._sync_mode_combo()

    # -- the API ----------------------------------------------------------------

    def select(self, value: int) -> None:
        """Makes `value`'s row current: switches to its category, drops a class
        filter that excludes it, ticks show-hidden for a hidden row (a pinned one
        needs no tick), and lands on it even when the text filter hides it.
        In the favorites view a favorite stays there, with no box ticked.
        Unknown value: no-op."""
        entry = self._by_id.get(value)
        if entry is None:
            return
        in_view = self.in_favorites_view() and value in self._items
        if not in_view and entry.category != self._category:
            self._set_category(entry.category)
        class_id = self.class_combo.currentData()
        if class_id is not None and class_id != entry.class_id:
            self.class_combo.setCurrentIndex(0)
        if not in_view and entry.hidden and value != self._pinned and not self.show_hidden_checkbox.isChecked():
            self.show_hidden_checkbox.setChecked(True)
        item = self._items.get(value)
        if item is not None:
            self.list.setCurrentItem(item)
            self.list.scrollToItem(item)

    def highlight(self, value: int | None) -> None:
        """Makes `value`'s row current with no signal, if it is listed and shown here;
        never switches category or ticks a box."""
        item = self._items.get(value)
        if item is None or item.isHidden() or value not in self._by_id:
            return
        self.list.blockSignals(True)
        self.list.setCurrentItem(item)
        self.list.blockSignals(False)
        self.list.scrollToItem(item)

    def pin(self, value: int | None) -> None:
        """Keeps `value`'s row out of the hidden box's filter (None: no row), so a
        field whose value is a hidden object can still open on it and keep it."""
        self._pinned = value
        self._on_hidden_toggled()

    def focus_first_row(self) -> None:
        """Makes the first visible row current and gives the list focus. No visible row: no-op."""
        rows = (self.list.item(index) for index in range(self.list.count()))
        first = next((item for item in rows if not item.isHidden()), None)
        if first is None:
            return
        self.list.setCurrentItem(first)
        self.list.setFocus()

    def clear_current(self) -> None:
        """No current row, and no current_changed: a row left current while
        hidden would come back looking picked, and a click on it emits nothing."""
        self.list.blockSignals(True)
        self.list.setCurrentRow(-1)
        self.list.blockSignals(False)

    def current_value(self) -> int | None:
        """The current row's value, or None for no row, a row a filter hides, or an unknown id."""
        item = self.list.currentItem()
        if item is None or item.isHidden():
            return None
        value = item.data(_VALUE_ROLE)
        return value if value in self._by_id else None

    def mode(self) -> str:
        """The mode shown: the chosen one, or "names" while no install can supply thumbnails."""
        return self._mode if self._has_install() else "names"

    def entry(self, value: int):
        """`value`'s catalog entry, random rows included, or None for an id this
        dataset lacks. Its name is the standard one, not the owner's."""
        return self._base_by_id.get(value)

    # -- favorites (GH #195) ------------------------------------------------------

    def favorites(self) -> list[int]:
        """The stored favorite ids in the user's order; [] without a store."""
        return list(self._favorites_store[0]()) if self._favorites_store is not None else []

    def set_favorites(self, ids: list[int]) -> None:
        """Stores `ids` as the favorites, in that order (the store's one writer), then
        refreshes the stars and the view and emits favorites_changed."""
        if self._favorites_store is None:
            return
        self._favorites_store[1](list(ids))
        self._refresh_favorites()
        self.favorites_changed.emit()

    def add_favorite(self, value: int) -> None:
        if value not in self.favorites():
            self.set_favorites([*self.favorites(), value])

    def remove_favorite(self, value: int) -> None:
        if value in self.favorites():
            self.set_favorites([v for v in self.favorites() if v != value])

    def in_favorites_view(self) -> bool:
        return self._category == _FAVORITES

    def set_favorites_view(self, on: bool) -> None:
        """Lists only the favorites (on), or the category it left (off)."""
        if self.favorites_button is None or on == self.in_favorites_view():
            return
        self.favorites_button.setChecked(on)
        if on:
            self._last_category = self._category
            self._category = _FAVORITES
            # No category button is checked while the view is on.
            self._category_group.setExclusive(False)
            for button in self.category_buttons.values():
                button.setChecked(False)
            self._category_group.setExclusive(True)
            self._populate()
        else:
            self._set_category(self._last_category)

    def _on_favorites_clicked(self, checked: bool) -> None:
        self.set_favorites_view(checked)
        if not checked:
            self.favorites_view_left.emit()

    def _favorite_menu(self, value: int) -> QMenu:
        """The right-click menu for `value`'s row: Add or Remove (an unknown row is only ever a favorite)."""
        menu = QMenu(self)
        if value in self.favorites():
            action = menu.addAction("Remove from Favorites")
            action.triggered.connect(lambda _checked=False, v=value: self.remove_favorite(v))
        elif value in self._by_id:
            action = menu.addAction("Add to Favorites")
            action.triggered.connect(lambda _checked=False, v=value: self.add_favorite(v))
        return menu

    def _toggle_favorite(self, value: int) -> None:
        """A right-click with Settings > Advanced's "Adds or removes the star at once": no menu."""
        if value in self.favorites():
            self.remove_favorite(value)
        elif value in self._by_id:
            self.add_favorite(value)

    def _show_favorite_menu(self, pos: QPoint) -> None:
        item = self.list.itemAt(pos)
        if item is None or item.isHidden():
            return
        if settings.get_units_catalog_right_click() == "toggle":
            self._toggle_favorite(item.data(_VALUE_ROLE))
            return
        menu = self._favorite_menu(item.data(_VALUE_ROLE))
        if menu.actions():
            menu.exec_(self.list.viewport().mapToGlobal(pos))
        menu.deleteLater()

    def _refresh_favorites(self) -> None:
        """The stars and tooltips in place, or the whole favorites view while it is shown."""
        if self.in_favorites_view():
            self._populate()
            return
        favorites = set(self.favorites())
        for value, item in self._items.items():
            item.setData(FAVORITE_ROLE, value in favorites)
            item.setToolTip(row_tooltip(self._by_id[value], value in favorites))
        # A star changes a list row's width hint, so the layout is redone.
        self.list.doItemsLayout()

    def _opening_category(self, wanted: str) -> str:
        names = [category for category, _label in CATEGORIES]
        present = {entry.category for entry in self._entries}
        return next((category for category in (wanted, *names) if category in present), names[0])

    def _limit_categories(self) -> None:
        """A given catalog can lack categories: their buttons go dead, and with
        at most one category left the block goes."""
        present = {entry.category for entry in self._entries}
        for category, button in self.category_buttons.items():
            button.setEnabled(category in present)
        self.category_block.setVisible(len(present) > 1)

    @staticmethod
    def _load_entries() -> tuple[tuple, dict]:
        """The active game dataset's objects, plus GH #164's random rows (in Return
        of Rome too: its 83 and 293 are the same male and female villagers, GH #67)."""
        entries = tuple(object_catalog.objects()) + tuple(place_random.catalog_entries())
        return entries, {entry.id: entry for entry in entries}

    def reload_entries(self) -> None:
        """After a game dataset switch: rows rebuilt from the new dataset's catalog; the favorites view stays."""
        self._base_entries, self._base_by_id = self._load_entries()
        self._name_entries()
        self._excluded_rows = self._rows_excluded_by(self._excluded)
        self._last_category = self._opening_category(self._last_category)
        self.class_combo.setProperty("category", None)
        self._populate()

    def set_excluded(self, consts: frozenset[int]) -> None:
        """Hides every row for these consts from the list, the counts and the
        class combo (GH #32). A random pick is hidden only when all its choices are."""
        consts = frozenset(consts)
        if consts == self._excluded:
            return
        self._excluded = consts
        self._excluded_rows = self._rows_excluded_by(consts)
        self._fill_class_combo()
        self._apply_filter()
        self._update_counts()

    def _rows_excluded_by(self, consts: frozenset[int]) -> frozenset[int]:
        return frozenset(entry.id for entry in self._entries if consts.issuperset(place_random.choices(entry.id)))

    def refresh_thumbnails(self) -> None:
        """After an install change: icon modes on or off, icons cleared, the viewport re-requested."""
        self._sync_mode_combo()
        self._populate()

    def set_team_color(self, rgb: tuple[int, int, int]) -> None:
        """Refills the rows on screen; a row scrolled back into view refills from the memo."""
        if tuple(rgb) == self._team_rgb:
            return
        self._team_rgb = tuple(rgb)
        self._request_viewport()

    def set_civ(self, civ: int | None) -> None:
        """The Owner's portrait civ (GH #372): re-requests the rows on screen; a
        row scrolled back into view takes the new civ's from the memo. Rows are
        renamed (an Athenian Archer reads Bowman) only when the two civs' own
        names differ."""
        if civ == self._civ:
            return
        old, self._civ = self._civ, civ
        if owner_renames(old, civ):
            self._name_entries()
            self._populate()
            return
        self._request_viewport()

    def _name_entries(self) -> None:
        """_entries and _by_id from the loaded rows, each named as the owner's civ names it."""
        entries = tuple(owner_entries(self._base_entries, self._civ))
        self._entries, self._by_id = entries, {entry.id: entry for entry in entries}

    # -- category, class, filter -------------------------------------------------

    def _on_category_clicked(self, category: str) -> None:
        left = self.in_favorites_view()
        self._set_category(category)
        if left:
            self.favorites_view_left.emit()

    def _set_category(self, category: str) -> None:
        if self.favorites_button is not None:
            self.favorites_button.setChecked(False)
        self._category = category
        self._last_category = category
        self.category_buttons[category].setChecked(True)
        self._populate()

    def _on_filter_changed(self) -> None:
        self._apply_filter()
        self._update_counts()

    def _on_hidden_toggled(self) -> None:
        self._fill_class_combo()
        self._apply_filter()
        self._update_counts()

    def _passes(self, entry, needle: str, show_hidden: bool, *, favorite: bool = False) -> bool:
        matches = not needle or needle in entry.search_text or needle in entry.name.lower()
        return self._listed(entry, show_hidden, favorite=favorite) and matches

    def _listed(self, entry, show_hidden: bool, *, favorite: bool = False) -> bool:
        """Whether a row passes the hidden box and set_excluded(), before any text or class filter.
        `favorite`: a favorites-view row, which the hidden box never hides (you chose it)."""
        shown = show_hidden or favorite or not entry.hidden or entry.id == self._pinned
        return shown and entry.id not in self._excluded_rows

    def _list_entries(self) -> list[tuple[int, object]]:
        """(value, entry or None) for the rows to list: the favorites in their order
        while the view is on, else the active category's entries by name."""
        if self.in_favorites_view():
            return [(value, self._by_id.get(value)) for value in self.favorites()]
        rows = sorted((e for e in self._entries if e.category == self._category), key=lambda e: e.name)
        return [(entry.id, entry) for entry in rows]

    def _update_counts(self) -> None:
        """Each button counts the rows passing the text filter and the hidden
        box, never the class combo, so a hit elsewhere is visible."""
        needle = self.filter_edit.text().strip().lower()
        show_hidden = self.show_hidden_checkbox.isChecked()
        counts = Counter(entry.category for entry in self._entries if self._passes(entry, needle, show_hidden))
        for category, label in CATEGORIES:
            self.category_buttons[category].setText(f"{label} ({counts[category]})")

    def _fill_class_combo(self) -> None:
        """"All classes", then each class in the listed rows (after the hidden box
        and set_excluded()) by count, most first. Keeps the chosen class if it is still there."""
        show_hidden = self.show_hidden_checkbox.isChecked()
        favorite = self.in_favorites_view()
        counts = Counter(
            entry.class_id
            for _value, entry in self._list_entries()
            if entry is not None and self._listed(entry, show_hidden, favorite=favorite)
        )
        chosen = self.class_combo.currentData()
        self.class_combo.blockSignals(True)
        self.class_combo.clear()
        self.class_combo.addItem(f"{_ALL_CLASSES} ({sum(counts.values())})", None)
        for class_id, count in sorted(counts.items(), key=lambda pair: (-pair[1], class_label(pair[0]))):
            self.class_combo.addItem(f"{class_label(class_id)} ({count})", class_id)
        index = self.class_combo.findData(chosen) if chosen is not None else 0
        self.class_combo.setCurrentIndex(max(index, 0))
        self.class_combo.blockSignals(False)

    def _apply_filter(self) -> None:
        needle = self.filter_edit.text().strip().lower()
        show_hidden = self.show_hidden_checkbox.isChecked()
        class_id = self.class_combo.currentData()
        favorite = self.in_favorites_view()
        for value, item in self._items.items():
            entry = self._by_id.get(value)
            if entry is None:
                # An unknown id: matched by its label's text and id, never by a class.
                shown = class_id is None and (not needle or needle in unknown_label(value).lower())
            else:
                shown = self._passes(entry, needle, show_hidden, favorite=favorite) and (
                    class_id is None or entry.class_id == class_id
                )
            item.setHidden(not shown)
        # Reorder only in the favorites view, and only with every row shown.
        set_drag_reorder(self.list, favorite and not any(item.isHidden() for item in self._items.values()))
        self._request_viewport()

    # -- the list -----------------------------------------------------------------

    def _populate(self) -> None:
        """Rebuilds the listed rows in the current mode, keeping the current row."""
        current = self.list.currentItem()
        keep = None if current is None else current.data(_VALUE_ROLE)
        self.class_combo.blockSignals(True)
        if self.class_combo.count() == 0 or self.class_combo.property("category") != self._category:
            self.class_combo.setProperty("category", self._category)
            self.class_combo.clear()
        self.class_combo.blockSignals(False)
        self._fill_class_combo()
        configure_list(self.list, self.mode())
        self.list.blockSignals(True)
        self._items = fill_rows(self.list, self._list_entries(), self.mode(), set(self.favorites()))
        if keep in self._items:
            self.list.setCurrentItem(self._items[keep])
        self.list.blockSignals(False)
        self._apply_filter()
        self._update_counts()

    def _request_viewport(self) -> None:
        px = icon_px(self.mode())
        if px is None or not self._items:
            return
        wanted = []
        # An unknown id never reaches the loader, which would decode it on the GUI thread.
        for value in (v for v in visible_values(self.list) if v in self._by_id):
            hit, icon = self._loader.cached(thumb_const(value), px, self._civ)
            if hit:
                self._set_icon(value, icon, px)
            else:
                wanted.append(thumb_const(value))
        if wanted:
            self._loader.request(wanted, px, client=self, civ=self._civ)

    def _on_thumb_ready(self, const: int, px: int, icon, civ: int | None = None) -> None:
        # Another owner's portrait (a card's, or from before an Owner switch) never lands on a row.
        if not thumb_fits(const, px, civ, icon_px(self.mode()), self._civ):
            return
        for value in (const, *place_random.RANDOM_PICKS):
            if value in self._items and value in self._by_id and thumb_const(value) == const:
                self._set_icon(value, icon, px)

    def _set_icon(self, value: int, icon, px: int) -> None:
        name = self._by_id[value].name
        self._items[value].setIcon(
            row_icon(self._loader, value, icon, px, self._civ, self._team_rgb, name, self.mode() == "grid")
        )

    def _on_activated(self, item) -> None:
        if item is None or item.isHidden() or item.data(_VALUE_ROLE) not in self._by_id:
            return
        self.activated.emit(item.data(_VALUE_ROLE))

    def _on_clicked(self, item) -> None:
        if item is None or item.isHidden() or item.data(_VALUE_ROLE) not in self._by_id:
            return
        self.picked.emit(item.data(_VALUE_ROLE))

    def _on_key_moved(self) -> None:
        value = self.current_value()
        if value is not None:
            self.picked.emit(value)

    # -- display modes -------------------------------------------------------------

    def _has_install(self) -> bool:
        return asset_source.get_install_path() is not None

    def _choose_mode(self, mode: str) -> None:
        if mode != self._mode:
            self._mode = mode
            self._save_mode(mode)
        self._sync_mode_combo()
        self._populate()

    def _sync_mode_combo(self) -> None:
        """Without an install the icon modes are disabled and the list shows names."""
        sync_mode_combo(self.display_combo, self.mode(), self._has_install())


def sync_mode_combo(combo: QComboBox, mode: str, install: bool) -> None:
    """A display dropdown showing `mode`, its icon modes disabled without an install."""
    model = combo.model()
    for index in range(combo.count()):
        mode_id = combo.itemData(index)
        enabled = install or mode_id == "names"
        model.item(index).setEnabled(enabled)
        combo.setItemData(index, _MODES[mode_id][1] if enabled else _NO_INSTALL_TIP, Qt.ToolTipRole)
    combo.setCurrentIndex(combo.findData(mode))
    # The closed dropdown's own, as the Icons/Names/Grid buttons had: why only Names is offered.
    combo.setToolTip(_MODES[mode][1] if install else _NO_INSTALL_COMBO_TIP)
