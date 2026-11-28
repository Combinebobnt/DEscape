"""The Favorite Objects window (GH #195): the Units catalog's favorites, one click to place.

Shaped like clipboard_dialog.ClipboardHistoryDialog: non-modal (show(), never
exec_()), it imports no model and reports intent through its callbacks. It is
handed (id, CatalogEntry or None) rows in the user's order; None is an id the
active dataset lacks, shown greyed as catalog_browser's "Unknown object (id)",
selectable and removable but never picked. Each row is named as the Owner's
civ names it (an Athenian Archer reads Bowman), renamed on an Owner switch only
when the two civs' own names differ, as the catalog does.

Its list is a catalog_browser.CatalogList with the catalog's delegate, so the
rows, the star, the right-press rule and the drag reorder are the catalog's.
It asks the panel's shared ThumbLoader for thumbnails as its own client: the
loader keeps one pending list per client, so sharing the catalog's would
cancel each other's requests.

A focused top-level dialog does not fire the main window's WindowShortcut
actions (clipboard_dialog.py's docstring), so Delete in its list removes a
favorite and never reaches the map's Delete.
"""

from __future__ import annotations

from collections.abc import Callable

from PyQt5.QtCore import QEvent, QPoint, Qt
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QListWidgetItem,
    QMenu,
    QPushButton,
    QVBoxLayout,
)

from descape import asset_source, catalog_browser, settings
from descape.catalog_browser import CatalogDelegate, CatalogList
from descape.terrain_palette import PORTRAIT_COLORS

_VALUE_ROLE = Qt.UserRole


class UnitFavoritesDialog(QDialog):
    EMPTY_HINT = "Right-click an object in the Units catalog to add it."

    def __init__(
        self,
        parent=None,
        *,
        loader,
        on_pick: Callable[[int], None] | None = None,
        on_reorder: Callable[[list[int]], None] | None = None,
        on_remove: Callable[[int], None] | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Favorite Objects")
        self.resize(300, 470)
        self.setMinimumSize(200, 280)
        self._loader = loader
        self._loader.ready.connect(self._on_thumb_ready)
        self._on_pick = on_pick or (lambda _value: None)
        self._on_reorder = on_reorder or (lambda _ids: None)
        self._on_remove = on_remove or (lambda _value: None)
        self._base_rows: list[tuple[int, object]] = []
        self._rows: list[tuple[int, object]] = []  # _base_rows named by the owner's civ
        self._items: dict[int, QListWidgetItem] = {}
        self._team_rgb: tuple[int, int, int] = PORTRAIT_COLORS[0]
        self._civ: int | None = None
        self._mode = settings.get_units_favorites_mode()
        self._pick_enabled = True
        self._disabled_hint = ""

        self.display_combo = catalog_browser.mode_combo()
        self.display_combo.activated.connect(lambda index: self._choose_mode(self.display_combo.itemData(index)))
        top_row = QHBoxLayout()
        top_row.addWidget(QLabel("View:"))
        top_row.addWidget(self.display_combo)
        top_row.addStretch(1)

        self.list = CatalogList()
        # ClickFocus: a click picks and hands the keys back to the map; Delete needs the list focused.
        self.list.setFocusPolicy(Qt.ClickFocus)
        self.list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.list.setTextElideMode(Qt.ElideNone)
        self.list.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.list.setItemDelegate(CatalogDelegate(self.list))
        self.list.itemClicked.connect(self._on_clicked)
        self.list.currentItemChanged.connect(lambda *_: self._update_buttons())
        self.list.verticalScrollBar().valueChanged.connect(lambda _value: self._request_viewport())
        self.list.horizontalScrollBar().valueChanged.connect(lambda _value: self._request_viewport())
        self.list.viewport_changed.connect(self._request_viewport)
        self.list.reordered.connect(self._on_reorder)
        self.list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.list.customContextMenuRequested.connect(self._show_remove_menu)
        self.list.installEventFilter(self)

        self.hint = QLabel()
        self.hint.setWordWrap(True)

        self.remove_button = QPushButton("Remove")
        self.remove_button.setToolTip("Remove the highlighted object from your favorites")
        self.remove_button.clicked.connect(lambda _checked=False: self._remove_current())
        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.close)
        buttons.button(QDialogButtonBox.Close).clicked.connect(self.close)
        button_row = QHBoxLayout()
        button_row.addWidget(self.remove_button)
        button_row.addStretch(1)
        button_row.addWidget(buttons)

        layout = QVBoxLayout(self)
        layout.addLayout(top_row)
        layout.addWidget(self.list, stretch=1)
        layout.addWidget(self.hint)
        layout.addLayout(button_row)

        self.set_entries([])

    # -- the API --------------------------------------------------------------

    def set_entries(self, rows) -> None:
        """Refills from (id, CatalogEntry or None) rows in the user's order, each named as the
        owner's civ names it. The highlighted row is kept by id while it is listed, and the
        visible rows' thumbnails are re-requested."""
        self._base_rows = list(rows)
        self._fill()

    def _fill(self) -> None:
        current = self.list.currentItem()
        keep = None if current is None else current.data(_VALUE_ROLE)
        values = [value for value, _entry in self._base_rows]
        entries = catalog_browser.owner_entries((entry for _value, entry in self._base_rows), self._civ)
        self._rows = list(zip(values, entries, strict=True))
        catalog_browser.sync_mode_combo(self.display_combo, self.mode(), self._has_install())
        catalog_browser.configure_list(self.list, self.mode())
        self.list.blockSignals(True)
        self._items = catalog_browser.fill_rows(self.list, self._rows, self.mode(), set(self.values()))
        if keep in self._items:
            self.list.setCurrentItem(self._items[keep])
        self.list.blockSignals(False)
        self._sync_state()
        self._request_viewport()

    def set_team_color(self, rgb: tuple[int, int, int]) -> None:
        if tuple(rgb) == self._team_rgb:
            return
        self._team_rgb = tuple(rgb)
        self._request_viewport()

    def set_civ(self, civ: int | None) -> None:
        """The Owner's portrait civ (GH #372); the rows are renamed only when the two civs' own names differ."""
        if civ == self._civ:
            return
        old, self._civ = self._civ, civ
        if catalog_browser.owner_renames(old, civ):
            self._fill()
            return
        self._request_viewport()

    def set_pick_enabled(self, enabled: bool, hint: str = "") -> None:
        """Off while Place Unit is unavailable: the list greys out and `hint` says why."""
        self._pick_enabled = enabled
        self._disabled_hint = hint
        self._sync_state()

    def mode(self) -> str:
        """The mode shown: the chosen one, or "names" while no install can supply thumbnails."""
        return self._mode if self._has_install() else "names"

    def values(self) -> list[int]:
        return [value for value, _entry in self._rows]

    # -- state ----------------------------------------------------------------

    def _sync_state(self) -> None:
        self.list.setEnabled(self._pick_enabled)
        if not self._pick_enabled:
            text = self._disabled_hint
        elif not self._rows:
            text = self.EMPTY_HINT
        else:
            text = ""
        self.hint.setText(text)
        self.hint.setVisible(bool(text))
        catalog_browser.set_drag_reorder(self.list, self._pick_enabled and len(self._rows) > 1)
        self._update_buttons()

    def _update_buttons(self) -> None:
        self.remove_button.setEnabled(self._pick_enabled and self.list.currentItem() is not None)

    def _has_install(self) -> bool:
        return asset_source.get_install_path() is not None

    def _choose_mode(self, mode: str) -> None:
        if mode != self._mode:
            self._mode = mode
            settings.set_units_favorites_mode(mode)
        self._fill()

    # -- picking and removing ---------------------------------------------------

    def _on_clicked(self, item) -> None:
        if not self._pick_enabled or item is None:
            return
        value = item.data(_VALUE_ROLE)
        if self._entry(value) is not None:
            self._on_pick(value)

    def _entry(self, value: int):
        return next((entry for v, entry in self._rows if v == value), None)

    def _remove_current(self) -> None:
        item = self.list.currentItem()
        if self._pick_enabled and item is not None:
            self._on_remove(item.data(_VALUE_ROLE))

    def _remove_menu(self, value: int) -> QMenu:
        menu = QMenu(self)
        action = menu.addAction("Remove from Favorites")
        action.triggered.connect(lambda _checked=False, v=value: self._on_remove(v))
        return menu

    def _show_remove_menu(self, pos: QPoint) -> None:
        item = self.list.itemAt(pos)
        if not self._pick_enabled or item is None:
            return
        menu = self._remove_menu(item.data(_VALUE_ROLE))
        menu.exec_(self.list.viewport().mapToGlobal(pos))
        menu.deleteLater()

    def eventFilter(self, watched, event) -> bool:  # Qt override
        # Delete with the list focused; accepted here, since on_remove rebuilds the list under the key event.
        if watched is self.list and event.type() == QEvent.KeyPress and event.key() == Qt.Key_Delete:
            self._remove_current()
            event.accept()
            return True
        return super().eventFilter(watched, event)

    # -- thumbnails -------------------------------------------------------------

    def showEvent(self, event) -> None:  # Qt override
        super().showEvent(event)
        self._request_viewport()

    def hideEvent(self, event) -> None:  # Qt override
        super().hideEvent(event)
        # Drop what this window still has queued, so a closed window costs no decodes.
        self._loader.request([], 0, client=self)

    def _request_viewport(self) -> None:
        px = catalog_browser.icon_px(self.mode())
        # Hidden (closed): nothing to fill, and the shared loader serves the catalog alone.
        if px is None or not self._items or not self.isVisible():
            return
        known = {value for value, entry in self._rows if entry is not None}
        wanted = []
        for value in (v for v in catalog_browser.visible_values(self.list) if v in known):
            const = catalog_browser.thumb_const(value)
            hit, icon = self._loader.cached(const, px, self._civ)
            if hit:
                self._set_icon(value, icon, px)
            else:
                wanted.append(const)
        if wanted:
            self._loader.request(wanted, px, client=self, civ=self._civ)

    def _on_thumb_ready(self, const: int, px: int, icon, civ: int | None = None) -> None:
        if not catalog_browser.thumb_fits(const, px, civ, catalog_browser.icon_px(self.mode()), self._civ):
            return
        for value, entry in self._rows:
            if entry is not None and catalog_browser.thumb_const(value) == const:
                self._set_icon(value, icon, px)

    def _set_icon(self, value: int, icon, px: int) -> None:
        entry = self._entry(value)
        grid = self.mode() == "grid"
        self._items[value].setIcon(
            catalog_browser.row_icon(self._loader, value, icon, px, self._civ, self._team_rgb, entry.name, grid)
        )
