"""CatalogBrowserView: the Units panel's object picker (GH #68, GH #6).

Like the in-game editor's object menus: a 2x2 block of category buttons
(Units, Buildings, Heroes, Other), a flat list of the active category, a
class combo that narrows it, a name-or-id filter and three display modes
(icon and name rows, text rows, an icon-only grid). ValuePickerView stays the
picker for every other consumer; this one reads object_catalog.objects()
directly, since it needs each entry's class, plus place_random's pseudo rows
(GH #164).

The API the panel and its tests use is ValuePickerView's: `activated(int)`,
`current_changed(object)`, select(), current_value(), `filter_edit`,
`show_hidden_checkbox`; plus `list`, `category_buttons`, `class_combo`,
`display_buttons` and refresh_thumbnails().

Rows of the active category stay in the list and are hidden by the filter,
the class combo and the editor-hidden box, so select() can make a filtered-out
row current the way ValuePickerView's tree does (UnitsPanel.pick_object relies
on that). Thumbnails come from the panel's shared catalog_thumbs.ThumbLoader,
requested for the rows in the viewport only.
"""

from __future__ import annotations

from collections import Counter

from AoE2ScenarioParser.datasets.trigger_lists import ObjectClass
from PyQt5.QtCore import QPoint, QRect, QSize, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QIcon, QPainter, QPixmap
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
    QPushButton,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from descape import asset_source, catalog_thumbs, game_dataset, object_catalog, place_random, settings
from descape.constant_picker import HIDDEN_LABEL
from descape.terrain_palette import PLAYER_COLORS

# (object_catalog category, button label), in the in-game editor's order, two to a row.
CATEGORIES = (("Units", "Units"), ("Buildings", "Buildings"), ("Heroes", "Heroes"), ("Others", "Other"))
# Display mode -> (button label, tooltip).
_MODES = {
    "icons": ("Icons", "Icon and name rows"),
    "names": ("Names", "Name rows only"),
    "grid": ("Grid", "Icons only, name and Unit ID on hover"),
}
_NO_INSTALL_TIP = "Needs the game install (Settings), which supplies the portraits and sprites"
_LIST_ICON_PX = 32
_GRID_ICON_PX = 48
_GRID_CELL_PAD = 8
_ALL_CLASSES = "All classes"
# The class combo's width in characters, so it and the three mode buttons fit MIN_USEFUL_WIDTH.
_CLASS_COMBO_CHARS = 8
_VALUE_ROLE = Qt.UserRole


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


def _thumb_const(value: int) -> int:
    """The const whose thumbnail a row shows: a random pick shows its preview const's."""
    return place_random.preview_const(value) if place_random.is_random(value) else value


class _CatalogList(QListWidget):
    """Scroll and resize both change which rows are on screen."""

    viewport_changed = pyqtSignal()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self.viewport_changed.emit()


class CatalogBrowserView(QWidget):
    activated = pyqtSignal(int)  # itemActivated: double-click and Enter, in every mode
    current_changed = pyqtSignal(object)  # int | None

    def __init__(self, loader: catalog_thumbs.ThumbLoader | None = None, parent: QWidget | None = None):
        super().__init__(parent)
        self._loader = loader if loader is not None else catalog_thumbs.ThumbLoader(self)
        self._loader.ready.connect(self._on_thumb_ready)
        self._entries, self._by_id = self._load_entries()
        self._items: dict[int, QListWidgetItem] = {}
        self._category = CATEGORIES[0][0]
        self._mode = settings.get_units_catalog_mode()
        # Behind a portrait's player-colour region: the Owner combo's, pushed by set_team_color().
        self._team_rgb: tuple[int, int, int] = PLAYER_COLORS[0]

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.category_buttons: dict[str, QPushButton] = {}
        self._category_group = QButtonGroup(self)
        self._category_group.setExclusive(True)
        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(2)
        for index, (category, label) in enumerate(CATEGORIES):
            button = QPushButton(label)
            button.setCheckable(True)
            button.setFocusPolicy(Qt.NoFocus)
            button.setProperty("label", label)
            button.clicked.connect(lambda _checked, c=category: self._set_category(c))
            self._category_group.addButton(button)
            self.category_buttons[category] = button
            grid.addWidget(button, index // 2, index % 2)
        self.category_buttons[self._category].setChecked(True)
        layout.addLayout(grid)

        class_row = QHBoxLayout()
        class_row.setContentsMargins(0, 0, 0, 0)
        self.class_combo = QComboBox()
        self.class_combo.setFocusPolicy(Qt.ClickFocus)
        # Sized by a character count, not its longest class; the popup shows each in full.
        self.class_combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.class_combo.setMinimumContentsLength(_CLASS_COMBO_CHARS)
        self.class_combo.currentIndexChanged.connect(lambda _index: self._apply_filter())
        class_row.addWidget(self.class_combo, stretch=1)
        # Tool buttons: a push button's style minimum (about 80 px each) would set the pane's floor.
        self.display_buttons: dict[str, QToolButton] = {}
        self._mode_group = QButtonGroup(self)
        self._mode_group.setExclusive(True)
        for mode, (label, tip) in _MODES.items():
            button = QToolButton()
            button.setText(label)
            button.setCheckable(True)
            button.setFocusPolicy(Qt.NoFocus)
            button.setToolTip(tip)
            button.clicked.connect(lambda _checked, m=mode: self._choose_mode(m))
            self._mode_group.addButton(button)
            self.display_buttons[mode] = button
            class_row.addWidget(button)
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

        self.list = _CatalogList()
        # ClickFocus: MapView's arrow-key nudge needs the map to keep focus (units_panel.py).
        self.list.setFocusPolicy(Qt.ClickFocus)
        self.list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.list.setTextElideMode(Qt.ElideNone)
        self.list.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.list.itemActivated.connect(self._on_activated)
        self.list.currentItemChanged.connect(lambda *_: self.current_changed.emit(self.current_value()))
        self.list.verticalScrollBar().valueChanged.connect(lambda _value: self._request_viewport())
        self.list.horizontalScrollBar().valueChanged.connect(lambda _value: self._request_viewport())
        self.list.viewport_changed.connect(self._request_viewport)
        layout.addWidget(self.list, stretch=1)

        self._populate()
        self._sync_mode_buttons()

    # -- the API ----------------------------------------------------------------

    def select(self, value: int) -> None:
        """Makes `value`'s row current: switches to its category, drops a class
        filter that excludes it, ticks show-hidden for a hidden row, and lands
        on it even when the text filter hides it. Unknown value: no-op."""
        entry = self._by_id.get(value)
        if entry is None:
            return
        if entry.category != self._category:
            self._set_category(entry.category)
        class_id = self.class_combo.currentData()
        if class_id is not None and class_id != entry.class_id:
            self.class_combo.setCurrentIndex(0)
        if entry.hidden and not self.show_hidden_checkbox.isChecked():
            self.show_hidden_checkbox.setChecked(True)
        item = self._items.get(value)
        if item is not None:
            self.list.setCurrentItem(item)
            self.list.scrollToItem(item)

    def current_value(self) -> int | None:
        """The current row's value, or None for no row or a row a filter hides."""
        item = self.list.currentItem()
        if item is None or item.isHidden():
            return None
        return item.data(_VALUE_ROLE)

    def mode(self) -> str:
        """The mode shown: the chosen one, or "names" while no install can supply thumbnails."""
        return self._mode if self._has_install() else "names"

    @staticmethod
    def _load_entries() -> tuple[tuple, dict]:
        """The active game dataset's objects, plus GH #164's random rows, which
        name AoE2 consts and so are left out under Return of Rome (GH #67)."""
        randoms = () if game_dataset.is_ror() else tuple(place_random.catalog_entries())
        entries = tuple(object_catalog.objects()) + randoms
        return entries, {entry.id: entry for entry in entries}

    def reload_entries(self) -> None:
        """After a game dataset switch: rows rebuilt from the new dataset's catalog."""
        self._entries, self._by_id = self._load_entries()
        self.class_combo.setProperty("category", None)
        self._populate()

    def refresh_thumbnails(self) -> None:
        """After an install change: icon modes on or off, icons cleared, the viewport re-requested."""
        self._sync_mode_buttons()
        self._populate()

    def set_team_color(self, rgb: tuple[int, int, int]) -> None:
        """Refills the rows on screen; a row scrolled back into view refills from the memo."""
        if tuple(rgb) == self._team_rgb:
            return
        self._team_rgb = tuple(rgb)
        self._request_viewport()

    # -- category, class, filter -------------------------------------------------

    def _set_category(self, category: str) -> None:
        self._category = category
        self.category_buttons[category].setChecked(True)
        self._populate()

    def _on_filter_changed(self) -> None:
        self._apply_filter()
        self._update_counts()

    def _on_hidden_toggled(self) -> None:
        self._fill_class_combo()
        self._apply_filter()
        self._update_counts()

    def _passes(self, entry, needle: str, show_hidden: bool) -> bool:
        return (show_hidden or not entry.hidden) and (not needle or needle in entry.search_text)

    def _update_counts(self) -> None:
        """Each button counts the rows passing the text filter and the hidden
        box, never the class combo, so a hit elsewhere is visible."""
        needle = self.filter_edit.text().strip().lower()
        show_hidden = self.show_hidden_checkbox.isChecked()
        counts = Counter(entry.category for entry in self._entries if self._passes(entry, needle, show_hidden))
        for category, label in CATEGORIES:
            self.category_buttons[category].setText(f"{label} ({counts[category]})")

    def _fill_class_combo(self) -> None:
        """"All classes", then each class in the active category (after the
        hidden box) by count, most first. Keeps the chosen class if it is still there."""
        show_hidden = self.show_hidden_checkbox.isChecked()
        counts = Counter(
            entry.class_id
            for entry in self._entries
            if entry.category == self._category and (show_hidden or not entry.hidden)
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
        for value, item in self._items.items():
            entry = self._by_id[value]
            shown = self._passes(entry, needle, show_hidden) and (class_id is None or entry.class_id == class_id)
            item.setHidden(not shown)
        self._request_viewport()

    # -- the list -----------------------------------------------------------------

    def _populate(self) -> None:
        """Rebuilds the active category's rows in the current mode, keeping the current row."""
        current = self.list.currentItem()
        keep = None if current is None else current.data(_VALUE_ROLE)
        self.class_combo.blockSignals(True)
        if self.class_combo.count() == 0 or self.class_combo.property("category") != self._category:
            self.class_combo.setProperty("category", self._category)
            self.class_combo.clear()
        self.class_combo.blockSignals(False)
        self._fill_class_combo()
        self._configure_mode()
        self.list.blockSignals(True)
        self.list.clear()
        self._items = {}
        grid = self.mode() == "grid"
        px = self._icon_px()
        # A blank icon of the mode's size: a row has its full size before its thumbnail lands,
        # and a uniform-size grid does not size every cell from an icon-less first one.
        placeholder = None if px is None else self._placeholder(px)
        for entry in sorted((e for e in self._entries if e.category == self._category), key=lambda e: e.name):
            item = QListWidgetItem("" if grid else entry.name)
            item.setData(_VALUE_ROLE, entry.id)
            item.setToolTip(f"{entry.name} ({entry.id})" if entry.id >= 0 else entry.name)
            if placeholder is not None:
                item.setIcon(placeholder)
            self.list.addItem(item)
            self._items[entry.id] = item
        if keep in self._items:
            self.list.setCurrentItem(self._items[keep])
        self.list.blockSignals(False)
        self._apply_filter()
        self._update_counts()

    def _configure_mode(self) -> None:
        mode = self.mode()
        if mode == "grid":
            self.list.setViewMode(QListView.IconMode)
            self.list.setIconSize(QSize(_GRID_ICON_PX, _GRID_ICON_PX))
            self.list.setGridSize(QSize(_GRID_ICON_PX + _GRID_CELL_PAD, _GRID_ICON_PX + _GRID_CELL_PAD))
            self.list.setResizeMode(QListView.Adjust)
            self.list.setMovement(QListView.Static)
            self.list.setWrapping(True)
            self.list.setUniformItemSizes(True)
        else:
            self.list.setViewMode(QListView.ListMode)
            self.list.setGridSize(QSize())
            self.list.setWrapping(False)
            self.list.setMovement(QListView.Static)
            self.list.setResizeMode(QListView.Fixed)
            self.list.setIconSize(QSize(_LIST_ICON_PX, _LIST_ICON_PX) if mode == "icons" else QSize())
            # Uniform sizes take every row's width from one row, clipping a longer name instead of scrolling to it.
            self.list.setUniformItemSizes(False)

    def _placeholder(self, px: int) -> QIcon:
        pixmap = QPixmap(px, px)
        pixmap.fill(Qt.transparent)
        return QIcon(pixmap)

    def _icon_px(self) -> int | None:
        return {"icons": _LIST_ICON_PX, "grid": _GRID_ICON_PX}.get(self.mode())

    def _visible_values(self) -> list[int]:
        """The rows on screen, top to bottom, found by indexAt from the
        viewport's top-left and walked forward until a row starts below it."""
        viewport = self.list.viewport().rect()
        first = self.list.indexAt(QPoint(2, 2))
        row = first.row() if first.isValid() else 0
        values = []
        for index in range(max(row - 1, 0), self.list.count()):
            item = self.list.item(index)
            if item.isHidden():
                continue
            rect = self.list.visualItemRect(item)
            if rect.top() > viewport.bottom():
                break
            if rect.intersects(viewport):
                values.append(item.data(_VALUE_ROLE))
        return values

    def _request_viewport(self) -> None:
        px = self._icon_px()
        if px is None or not self._items:
            return
        wanted = []
        for value in self._visible_values():
            hit, icon = self._loader.cached(_thumb_const(value), px)
            if hit:
                self._set_icon(value, icon, px)
            else:
                wanted.append(_thumb_const(value))
        if wanted:
            self._loader.request(wanted, px, client=self)

    def _on_thumb_ready(self, const: int, px: int, icon) -> None:
        if px != self._icon_px():
            return
        for value in (const, *place_random.RANDOM_PICKS):
            if value in self._items and _thumb_const(value) == const:
                self._set_icon(value, icon, px)

    def _set_icon(self, value: int, icon, px: int) -> None:
        item = self._items[value]
        icon = catalog_thumbs.team_filled(icon, self._loader.source(_thumb_const(value), px), self._team_rgb, px)
        if place_random.is_random(value):
            icon = _random_badge(icon, px)
        if icon is None and self.mode() == "grid":
            icon = _glyph(self._by_id[value].name, px)
        if icon is not None:
            item.setIcon(icon)

    def _on_activated(self, item) -> None:
        if item is None or item.isHidden():
            return
        self.activated.emit(item.data(_VALUE_ROLE))

    # -- display modes -------------------------------------------------------------

    def _has_install(self) -> bool:
        return asset_source.get_install_path() is not None

    def _choose_mode(self, mode: str) -> None:
        if mode != self._mode:
            self._mode = mode
            settings.set_units_catalog_mode(mode)
        self._sync_mode_buttons()
        self._populate()

    def _sync_mode_buttons(self) -> None:
        """Without an install the icon modes are disabled and the list shows names."""
        install = self._has_install()
        for mode, button in self.display_buttons.items():
            icon_mode = mode != "names"
            button.setEnabled(install or not icon_mode)
            button.setToolTip(_MODES[mode][1] if install or not icon_mode else _NO_INSTALL_TIP)
        self.display_buttons[self.mode()].setChecked(True)
