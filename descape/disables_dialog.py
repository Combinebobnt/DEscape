"""DisablesDialog: the in-game editor's "Disable Objects" control -- three
two-pane movers (Buildings / Units / Techs), one per category, per player.

Dumb and callback-driven like VariablesDialog and every other widget in this
family: it never imports OptionsEditModel and never touches EditHistory. It
is handed the current 24 lists at construction, mutates only its own pending
copy of them, and reports the changed ones through one callback when the
user accepts.

**One callback for the whole session, not one per click.** The dialog is
modal, so under exec_() the main window's Ctrl+Z cannot fire: an
apply-immediately variant would leave the user unable to undo a wrong Add
without closing the dialog first, one Ctrl+Z per click. Pending state plus
accept/reject also follows MirrorDialog, this repo's accept/reject
precedent, and needs no window-owned instance -- construct per open, exec_(),
discard.

The Full List panes are ValuePickerView, reused verbatim for its filter box
and grouping. Check-state is deliberately NOT added to that widget: it backs
every other picker in the app, and a two-pane mover is this dialog's problem,
not a new mode for everyone else.

An id the Full List cannot produce still renders and still round-trips. Real
corpus files disable 621 ("Town Center") and 35 ("Battering Ram"), neither of
which object_catalog.objects() carries -- see descape/disables_fields.py.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from descape import constant_picker, disables_fields, object_catalog
from descape.players_panel import _swatch_icon
from descape.value_picker import PickerItem, ValuePickerView

# Tab order, and the in-game editor's own: Buildings, Units, Techs. Not
# disables_fields.CATEGORIES, which is the order the *bytes* are in.
_TAB_ORDER = ("buildings", "units", "techs")

_TAB_LABELS = {"buildings": "Buildings", "units": "Units", "techs": "Techs"}

# Which object_catalog category each tab's Full List draws from -- exact
# in-game parity. Heroes and Others are deliberately absent: no corpus file
# disables one, and the in-game control does not offer them.
_OBJECT_CATEGORY = {"buildings": "Buildings", "units": "Units"}


def _full_list_items(category: str) -> tuple[PickerItem, ...]:
    """The tab's pickable rows, flattened to one ungrouped list.

    `group=""` rather than the catalog's own category: a single-category
    list would otherwise render under one collapsible heading containing
    everything, which is a click to open and nothing to choose between.
    `hidden` is preserved, so the objects tabs keep their editor-hidden
    filter working.
    """
    if category == "techs":
        catalog = object_catalog.techs()
    else:
        wanted = _OBJECT_CATEGORY[category]
        catalog = tuple(e for e in object_catalog.objects() if e.category == wanted)
    return tuple(
        PickerItem(
            label=item.label, value=item.value, group="", hidden=item.hidden,
            search_text=item.search_text,
        )
        for item in constant_picker.items_for(catalog)
    )


def _label_for(category: str, id_: int, catalog) -> str:
    """How a Disabled-pane row renders.

    The catalog's own label first, so the two panes never disagree: with no
    install configured object_name() resolves through the .dat short code
    while the Full List shows the library enum name, and a mover whose two
    sides name the same thing differently is a bug report waiting to happen.
    Falls back to the id-space resolver, which is what covers the
    out-of-enum ids a real file carries.
    """
    name = object_catalog.name_for(catalog, id_)
    if not name:
        name = (
            object_catalog.tech_name(id_)
            if category == "techs"
            else object_catalog.object_name(id_)
        )
    return f"{name} ({id_})"


class _CategoryTab(QWidget):
    """One category's two-pane mover: Full List, Add/Remove, Disabled List."""

    def __init__(self, category: str, on_changed, parent=None):
        super().__init__(parent)
        self.category = category
        self._on_changed = on_changed
        self._catalog = (
            object_catalog.techs()
            if category == "techs"
            else tuple(
                e for e in object_catalog.objects() if e.category == _OBJECT_CATEGORY[category]
            )
        )

        # No sprite preview on the Techs tab, and no hidden toggle there
        # either. The preview would be actively wrong: object ids and tech
        # ids are separate spaces that collide, so looking a tech id up in
        # unit_sprites returns an unrelated unit's graphic. The checkbox
        # would merely be dead: gen_object_catalog.py emits no `hidden`
        # flag for a tech, so every row's is False and the box would filter
        # nothing. Omitting it is the fix, not a gap.
        is_techs = category == "techs"
        self.full_list = ValuePickerView(
            _full_list_items(category),
            hidden_label="" if is_techs else constant_picker.HIDDEN_LABEL,
            preview=None if is_techs else constant_picker.catalog_preview,
        )
        self.full_list.activated.connect(lambda *_: self._add_selected())

        self.add_button = QPushButton("Add →")
        self.add_button.clicked.connect(lambda *_: self._add_selected())
        self.remove_button = QPushButton("← Remove")
        self.remove_button.clicked.connect(lambda *_: self._remove_selected())

        self.disabled_list = QListWidget()
        self.disabled_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.disabled_list.itemActivated.connect(lambda *_: self._remove_selected())

        buttons = QVBoxLayout()
        buttons.addStretch(1)
        buttons.addWidget(self.add_button)
        buttons.addWidget(self.remove_button)
        buttons.addStretch(1)

        right = QVBoxLayout()
        right.setContentsMargins(0, 0, 0, 0)
        right.addWidget(QLabel("Disabled"))
        right.addWidget(self.disabled_list, stretch=1)
        right_host = QWidget()
        right_host.setLayout(right)

        left = QVBoxLayout()
        left.setContentsMargins(0, 0, 0, 0)
        left.addWidget(QLabel("Available"))
        left.addWidget(self.full_list, stretch=1)
        left_host = QWidget()
        left_host.setLayout(left)

        row = QHBoxLayout(self)
        row.addWidget(left_host, stretch=3)
        row.addLayout(buttons)
        row.addWidget(right_host, stretch=2)

    # -- population ------------------------------------------------------

    def show_ids(self, ids: Sequence[int]) -> None:
        """Repopulate the Disabled pane. Order is preserved as stored/entered
        -- the game does not appear to care, and keeping it is what lets an
        untouched list re-encode byte-identically."""
        self.disabled_list.clear()
        for id_ in ids:
            item = QListWidgetItem(_label_for(self.category, id_, self._catalog))
            item.setData(Qt.UserRole, id_)
            self.disabled_list.addItem(item)

    def disabled_ids(self) -> tuple[int, ...]:
        return tuple(
            self.disabled_list.item(i).data(Qt.UserRole)
            for i in range(self.disabled_list.count())
        )

    # -- mutation --------------------------------------------------------

    def _add_selected(self) -> None:
        value = self.full_list.current_value()
        if value is None or value in self.disabled_ids():
            return
        self._on_changed(self.category, (*self.disabled_ids(), value))

    def _remove_selected(self) -> None:
        doomed = {item.data(Qt.UserRole) for item in self.disabled_list.selectedItems()}
        if not doomed:
            return
        self._on_changed(self.category, tuple(i for i in self.disabled_ids() if i not in doomed))


class DisablesDialog(QDialog):
    """Modal Disabled Objects editor. Construct per open, exec_(), discard.

    `current` is {(category, player_id): ids} for all 24 lists, which the
    window builds from the model's pending values where it has them and the
    file's stored ones otherwise. `on_accept` is handed only the entries that
    actually differ from that starting state, so an OK that changed nothing
    pushes no undo record.
    """

    def __init__(
        self,
        current: Mapping[tuple[str, int], Sequence[int]],
        player_colors: Mapping[int, tuple[int, int, int]] | None = None,
        on_accept=None,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Disabled Objects")
        self.resize(720, 480)
        self._on_accept = on_accept or (lambda *args: None)
        self._original = {key: tuple(value) for key, value in current.items()}
        self._pending = dict(self._original)
        self._player_id = 1
        self._populating = False

        layout = QVBoxLayout(self)

        self.player_combo = QComboBox()
        for player_id in range(1, disables_fields.NUM_LIST_PLAYERS + 1):
            colors = player_colors or {}
            label = f"P{player_id}"
            if player_id in colors:
                self.player_combo.addItem(_swatch_icon(colors[player_id]), label)
            else:
                self.player_combo.addItem(label)
        self.player_combo.currentIndexChanged.connect(self._on_player_changed)
        layout.addWidget(self.player_combo)

        self.tabs = QTabWidget()
        self._tabs: dict[str, _CategoryTab] = {}
        for category in _TAB_ORDER:
            tab = _CategoryTab(category, self._changed)
            self._tabs[category] = tab
            self.tabs.addTab(tab, _TAB_LABELS[category])
        layout.addWidget(self.tabs, stretch=1)

        # No read-only mode. Both entry points are gated on
        # options_model.disables_write_supported(), so a file this could not
        # write never reaches here -- see ViewerWindow._disables_editable().
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self._accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)

        self._populate()

    # -- state -----------------------------------------------------------

    def _populate(self) -> None:
        """The single repopulate path -- construction, and every player
        change -- so the _populating guard covers both."""
        self._populating = True
        try:
            for category, tab in self._tabs.items():
                tab.show_ids(self._pending.get((category, self._player_id), ()))
        finally:
            self._populating = False

    def _on_player_changed(self, index: int) -> None:
        self._player_id = index + 1
        self._populate()

    def _changed(self, category: str, ids: Sequence[int]) -> None:
        """A tab reports its new list. The equality guard plus _populating is
        the house pattern from PlayersPanel._changed(): a repopulate must
        never read back as a user edit."""
        if self._populating:
            return
        key = (category, self._player_id)
        ids = tuple(ids)
        if self._pending.get(key, ()) == ids:
            return
        self._pending[key] = ids
        self._tabs[category].show_ids(ids)

    def _accept(self) -> None:
        changes = {
            key: value for key, value in self._pending.items() if value != self._original.get(key, ())
        }
        self._on_accept(changes)
        self.accept()

    # -- test accessors ---------------------------------------------------

    def current_player_id(self) -> int:
        return self._player_id

    def select_player(self, player_id: int) -> None:
        self.player_combo.setCurrentIndex(player_id - 1)

    def select_tab(self, category: str) -> None:
        self.tabs.setCurrentWidget(self._tabs[category])

    def full_list_for(self, category: str) -> ValuePickerView:
        return self._tabs[category].full_list

    def disabled_ids_for(self, category: str) -> tuple[int, ...]:
        return self._tabs[category].disabled_ids()

    def disabled_labels_for(self, category: str) -> tuple[str, ...]:
        widget = self._tabs[category].disabled_list
        return tuple(widget.item(i).text() for i in range(widget.count()))

    def add_id(self, category: str, id_: int) -> None:
        """What clicking Add does, without needing the Full List's selection
        to be driven first -- the tree is a real widget with real filter
        state, and a test that has to select in it is testing Qt."""
        tab = self._tabs[category]
        if id_ in tab.disabled_ids():
            return
        self._changed(category, (*tab.disabled_ids(), id_))

    def remove_id(self, category: str, id_: int) -> None:
        tab = self._tabs[category]
        self._changed(category, tuple(i for i in tab.disabled_ids() if i != id_))
