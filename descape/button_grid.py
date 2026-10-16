"""GH #169: the command panel a train or research location effect writes into.

A preview of the building's panel after the trigger fires, under the
`button_location` row of Change Train Location, Add Train Location and
Change Research Location. The chosen slot shows the effect's own unit, tech
or building. Clicking a cell
sets `button_location` through the form's own combo, so the write and its
undo step are the combo's.

bind() follows the property form's row widgets as FacetPreview does. What a
cell holds comes from command_buttons, which reads the .dat's train and
research locations. The chosen cell shows the button the game shows there:
the effect's own, or the slot's lower-id occupant that hides it
(command_buttons.shown_in_slot()), and its tooltip says which.

GH #230: the grid shows only what the effect player's civ can have
(civ_availability, through the PlayerContext bind()'s `context` returns for
the form's source_player), frames the cell the effect's own button lands in
(green when it shows, red when it won't or it covers a tech), and with "Show
all triggers' changes" applies the player's other location effects first
(train_locations.apply()), framing each cell they put a button in the same
way.
"""

from __future__ import annotations

from PyQt5.QtCore import QSize, Qt
from PyQt5.QtGui import QColor, QIcon, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from descape import asset_source, civ_availability, command_buttons, object_catalog, settings, train_locations
from descape.command_buttons import TECH, UNIT, building_name
from descape.train_locations import (  # noqa: F401 -- callers read the effect names here
    ADD_TRAIN,
    BUILDING_FIELD,
    CHANGE_RESEARCH,
    CHANGE_TRAIN,
    EFFECT_MOVERS,
    NO_CONTEXT,
    PLAYER_FIELD,
    SLOT_FIELD,
    PlayerContext,
)
from descape.value_picker import ValueLineEdit
from descape.viewer_common import pixmap_from_rgba

UNSET = -1
CELL_PX = 44
ICON_PX = 36
_DIMMED_OPACITY = 0.35
_MORE_LIMIT = 12
# The frame on a cell the effect (or, with the toggle, another effect) puts a button in:
# green when the button shows, red when it won't or it covers a tech.
FRAME_OK = "ok"
FRAME_PROBLEM = "problem"
FRAME_COLORS = {FRAME_OK: QColor(46, 170, 70), FRAME_PROBLEM: QColor(215, 48, 39)}
FRAME_WIDTH = 2
ALL_TRIGGERS_TEXT = "Show all triggers' changes"

PAGE_1, PAGE_2 = "page 1", "page 2"
_PAGE_TITLES = {
    PAGE_1: "Page 1",
    PAGE_2: "Page 2",
    command_buttons.ECONOMIC: "Economic",
    command_buttons.MILITARY: "Military",
}


def _name(id_: int, kind: str) -> str:
    return object_catalog.tech_name(id_) if kind == TECH else object_catalog.display_name(id_)


def listing(items) -> str:
    """One name per line, cut at _MORE_LIMIT."""
    lines = [_name(id_, kind) for id_, kind in items[:_MORE_LIMIT]]
    if len(items) > _MORE_LIMIT:
        lines.append(f"+{len(items) - _MORE_LIMIT} more")
    return "\n".join(lines)


def _sorts_after(data, value: int) -> bool:
    """Whether a combo row's data belongs after `value` (the "(differs)" row's None never does)."""
    return isinstance(data, int) and data > value


class _Cell(QToolButton):
    def __init__(self, row: int, col: int) -> None:
        super().__init__()
        self.row, self.col = row, col
        self.setFocusPolicy(Qt.NoFocus)
        self.setCheckable(True)
        self.setFixedSize(CELL_PX, CELL_PX)
        self.setIconSize(QSize(ICON_PX, ICON_PX))
        self.setToolButtonStyle(Qt.ToolButtonIconOnly)
        # What the cell shows, for tests and tooltips: (id, kind) items, base first.
        self.items: tuple[tuple[int, str], ...] = ()
        self.replaced: tuple[tuple[int, str], ...] = ()
        self.dimmed = False
        # The effect's button is chosen here but hidden by a lower-id occupant.
        self.blocked = False
        # "More"/"Cancel": a builder menu's own button sits here.
        self.reserved = ""
        # The effect's tech shares this slot with others and takes turns with them.
        self.shared = False
        # The effect's unit or tech is one the player's civ can't have: no button.
        self.unavailable = False
        # A button that won't show here (the civ can't have it), drawn dimmed in an otherwise empty cell.
        self.ghost: tuple[int, str] | None = None
        # FRAME_OK or FRAME_PROBLEM where the effect (or another one) puts a button, else "".
        self.frame = ""
        # Tooltip lines naming the other effects that changed this cell.
        self.notes: tuple[str, ...] = ()

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        if not self.frame:
            return
        painter = QPainter(self)
        pen = QPen(FRAME_COLORS[self.frame])
        pen.setWidth(FRAME_WIDTH)
        painter.setPen(pen)
        painter.drawRect(self.rect().adjusted(1, 1, -1, -1))
        painter.end()


class ButtonGrid(QWidget):
    """Header, page switch, the 3x5 cells and a footer, or a one-line hint."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFocusPolicy(Qt.NoFocus)
        self.header_label = QLabel()
        self.hint_label = QLabel()
        self.footer_label = QLabel()
        for label in (self.header_label, self.hint_label, self.footer_label):
            label.setWordWrap(True)
            label.setMinimumWidth(1)
            label.setFocusPolicy(Qt.NoFocus)

        self.page_buttons: dict[str, QToolButton] = {}
        self._page_group = QButtonGroup(self)
        self._page_group.setExclusive(True)
        self.page_row = QHBoxLayout()
        self.page_row.setContentsMargins(0, 0, 0, 0)
        self.page_row.setSpacing(2)
        for page, title in _PAGE_TITLES.items():
            button = QToolButton()
            button.setText(title)
            button.setCheckable(True)
            button.setFocusPolicy(Qt.NoFocus)
            button.clicked.connect(lambda _checked, p=page: self._show_page(p))
            self._page_group.addButton(button)
            self.page_buttons[page] = button
            self.page_row.addWidget(button)
        self.page_row.addStretch(1)
        # Below the footer: a view option, not a page control, and too wide for the page row.
        self.all_triggers_check = QCheckBox(ALL_TRIGGERS_TEXT)
        self.all_triggers_check.setFocusPolicy(Qt.NoFocus)
        self.all_triggers_check.setToolTip(
            "Also apply every other Change Train Location, Add Train Location and Change Research Location "
            "effect for this player, in trigger order, before this one."
        )
        self.all_triggers_check.setChecked(settings.get_button_grid_all_triggers())
        self.all_triggers_check.toggled.connect(self._all_triggers_toggled)

        self.cells: list[_Cell] = []
        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(2)
        for index in range(command_buttons.PAGE_SIZE):
            cell = _Cell(index // command_buttons.COLUMNS + 1, index % command_buttons.COLUMNS + 1)
            cell.clicked.connect(lambda _checked, c=cell: self._cell_clicked(c))
            grid.addWidget(cell, cell.row - 1, cell.col - 1)
            self.cells.append(cell)
        self.grid_host = QWidget()
        self.grid_host.setFocusPolicy(Qt.NoFocus)
        grid_row = QHBoxLayout(self.grid_host)
        grid_row.setContentsMargins(0, 0, 0, 0)
        grid_row.addLayout(grid)
        grid_row.addStretch(1)

        column = QVBoxLayout(self)
        column.setContentsMargins(0, 2, 0, 2)
        column.setSpacing(4)
        column.addWidget(self.header_label)
        column.addWidget(self.hint_label)
        column.addLayout(self.page_row)
        column.addWidget(self.grid_host)
        column.addWidget(self.footer_label)
        column.addWidget(self.all_triggers_check)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)

        self.page: str | None = None
        self._effect = ""
        self._fields: dict = {}
        self._read = None
        self._entry = None
        self._on_resize = None
        self._layout_key: tuple | None = None
        self._icons: dict[tuple[int, str], QPixmap | None] = {}
        self._context_for = None
        self.context: PlayerContext = NO_CONTEXT
        self._applied: train_locations.Applied | None = None

    def minimumSizeHint(self) -> QSize:
        return QSize(command_buttons.COLUMNS * (CELL_PX + 2), super().minimumSizeHint().height())

    # -- wiring --------------------------------------------------------------

    def bind(self, fields: dict, effect: str, read, entry=None, on_resize=None, context=None) -> None:
        """Follow the mover, building, slot and player rows of `fields`
        (field name -> row widget) for `effect`, one of EFFECT_MOVERS.
        `entry` is the entry the form shows; `on_resize` runs when the row
        changes shape; `context(player)` returns the PlayerContext for a
        source_player value (None: no civ filter, no other effects)."""
        mover, _kind = EFFECT_MOVERS[effect]
        self._effect = effect
        names = (mover, BUILDING_FIELD, SLOT_FIELD, PLAYER_FIELD)
        self._fields = {name: fields[name] for name in names if name in fields}
        self._read = read
        self._entry = entry
        self._on_resize = on_resize
        self._context_for = context
        for name in (mover, BUILDING_FIELD, PLAYER_FIELD):
            widget = self._fields.get(name)
            if isinstance(widget, ValueLineEdit):
                # Bound methods of this QObject: the connections die with it.
                widget.committed.connect(self._field_changed)
        for name in (SLOT_FIELD, PLAYER_FIELD):
            combo = self._fields.get(name)
            if isinstance(combo, QComboBox):
                combo.currentIndexChanged.connect(self._field_changed)
        self.page = None
        self.refresh()

    def detach(self) -> None:
        """Stop following: the rows are being torn down with this grid."""
        self._fields = {}
        self._entry = None
        self._on_resize = None
        self._context_for = None

    def _field_changed(self, *_args) -> None:
        self.page = None
        self.refresh()

    def _int(self, name: str) -> int:
        """A row's live value, UNSET when unset or unreadable."""
        widget = self._fields.get(name)
        if isinstance(widget, ValueLineEdit):
            value = widget.value()
        elif isinstance(widget, QComboBox):
            value = widget.currentData()
        else:
            value = None if self._entry is None else self._read(self._entry, name)
        return value if isinstance(value, int) and not isinstance(value, bool) else UNSET

    @property
    def kind(self) -> str:
        return EFFECT_MOVERS[self._effect][1]

    def mover(self) -> int:
        return self._int(EFFECT_MOVERS[self._effect][0])

    def building(self) -> int:
        return self._int(BUILDING_FIELD)

    def value(self) -> int:
        return self._int(SLOT_FIELD)

    def player(self) -> int:
        return self._int(PLAYER_FIELD)

    def _load_context(self) -> PlayerContext:
        if self._context_for is None:
            return NO_CONTEXT
        return self._context_for(self.player())

    @property
    def all_triggers(self) -> bool:
        return self.all_triggers_check.isChecked()

    @property
    def editable(self) -> bool:
        combo = self._fields.get(SLOT_FIELD)
        return isinstance(combo, QComboBox) and combo.isEnabled()

    # -- what the panel holds ------------------------------------------------

    def pages(self) -> tuple[str, ...]:
        """The pages this building's panel has: the builder's two menus, or
        two pages anywhere else (in game, page 2 works at a Dock and an Army Tent)."""
        if command_buttons.is_builder(self.building()):
            return (command_buttons.ECONOMIC, command_buttons.MILITARY)
        return (PAGE_1, PAGE_2)

    def _contents(self, page: str) -> dict[int, tuple[tuple[int, str], ...]]:
        """{button: (id, kind) items, base first} at the building, every page
        of a plain building, one menu of the villager."""
        building = self.building()
        villager_page = page if command_buttons.is_builder(building) else None
        keep = self.context.keep
        cells: dict[int, list[tuple[int, str]]] = {}
        for kind in (UNIT, TECH):
            for button, ids in command_buttons.occupants(building, kind, villager_page, keep).items():
                cells.setdefault(button, []).extend((id_, kind) for id_ in command_buttons.ordered(ids, kind))
        self._applied = None
        if self.all_triggers and self.context.edits:
            self._applied = train_locations.apply(cells, building, villager_page, self.context.edits, keep)
            cells = self._applied.cells
        return {button: tuple(items) for button, items in cells.items()}

    def _page_of_value(self) -> str:
        """The page the chosen slot sits on."""
        if command_buttons.is_builder(self.building()):
            mover_page = command_buttons.villager_page(self.mover()) if self.kind == UNIT else None
            return mover_page or command_buttons.ECONOMIC
        found = command_buttons.slot(self.value())
        return PAGE_2 if found is not None and found[0] == 2 else PAGE_1

    def _page_number(self, page: str) -> int:
        return 2 if page == PAGE_2 else 1

    def _clickable_page(self) -> bool:
        """The villager's other menu is for looking: a building's button
        always lands on the menu its interface_kind picks."""
        if not command_buttons.is_builder(self.building()) or self.kind != UNIT:
            return True
        mover_page = command_buttons.villager_page(self.mover())
        return mover_page is None or mover_page == self.page

    def _already_there(self) -> bool:
        """The mover already sits in the chosen slot, so nothing is replaced."""
        building = self.building()
        return (building, self.value()) in command_buttons.home_slots(self.mover(), self.kind)

    def _mover_available(self) -> bool:
        return self.mover() < 0 or self.context.keep(self.mover(), self.kind)

    # -- rendering -----------------------------------------------------------

    def _sync_choices(self) -> None:
        """Offer only command_buttons.slot_values() for the building in the
        slot combo. The current value keeps its row, so nothing is rewritten."""
        combo = self._fields.get(SLOT_FIELD)
        if not isinstance(combo, QComboBox):
            return
        wanted = command_buttons.slot_values(self.building())
        current = combo.currentData()
        if [combo.itemData(i) for i in range(combo.count()) if combo.itemData(i) != current] == [
            value for value in wanted if value != current
        ]:
            return
        combo.blockSignals(True)
        try:
            for index in reversed(range(combo.count())):
                value = combo.itemData(index)
                if value is not None and value != current and value not in wanted:
                    combo.removeItem(index)
            present = {combo.itemData(i) for i in range(combo.count())}
            for value in wanted:
                if value in present:
                    continue
                at = next((i for i in range(combo.count()) if _sorts_after(combo.itemData(i), value)), combo.count())
                combo.insertItem(at, f"{command_buttons.slot_label(value)} ({value})", value)
        finally:
            combo.blockSignals(False)

    def refresh(self) -> None:
        if not self._fields:
            return
        self.context = self._load_context()
        self._sync_choices()
        hint = self._hint()
        shown = not hint
        self.hint_label.setText(hint)
        self.hint_label.setVisible(bool(hint))
        self.header_label.setText(self._header())
        self.header_label.setVisible(self.building() >= 0)
        pages: tuple[str, ...] = ()
        if shown:
            pages = self.pages()
            if self.page not in pages:
                self.page = self._page_of_value()
            self._render_page()
            self.footer_label.setText(self._footer())
        for page, button in self.page_buttons.items():
            button.setVisible(len(pages) > 1 and page in pages)
            button.setChecked(page == self.page)
        self.all_triggers_check.setVisible(shown and self._context_for is not None)
        self.grid_host.setVisible(shown)
        self.footer_label.setVisible(shown and bool(self.footer_label.text()))
        key = (shown, len(pages) > 1, hint, self.footer_label.text(), self.header_label.text())
        if key != self._layout_key:
            self._layout_key = key
            self.updateGeometry()
            if self._on_resize is not None:
                self._on_resize()

    def _hint(self) -> str:
        what = "tech to research" if self.kind == TECH else "unit to train"
        if self.mover() < 0:
            return f"Pick the {what} to preview the building's buttons."
        building = self.building()
        if building < 0:
            where = "researched" if self.kind == TECH else "trained"
            return f"Pick where it is {where} to preview its buttons."
        if building in command_buttons.VILLAGER_IDS:
            return "A villager's build buttons move only when trained at the builder (118), not a villager."
        if not command_buttons.has_buttons(building) and not object_catalog.is_building(building):
            # A building with none still takes trained units (an Army Tent): it shows empty.
            return f"{building_name(building)} is not a building and has no command buttons in the game data."
        return ""

    def _header(self) -> str:
        building = self.building()
        if building < 0:
            return ""
        where = "Researched at" if self.kind == TECH else "Trained at"
        return f"{where}: {building_name(building)}{self._civ_note()}"

    def _civ_note(self) -> str:
        """" (Spanish)", or why nothing is filtered; "" without a context."""
        if self._context_for is None:
            return ""
        context = self.context
        if context.civ is not None:
            return f" ({context.civ_text})"
        if context.player == civ_availability.ALL_PLAYERS:
            return " (any civ: every player)"
        if not 1 <= context.player <= 8:
            return " (any civ: no player set)"
        civ = context.civ_text.lower() or "unknown"
        return f" (any civ: player {context.player}'s civ is {civ})"

    def _render_page(self) -> None:
        contents = self._contents(self.page)
        number = self._page_number(self.page)
        mover, kind = self.mover(), self.kind
        chosen = self.value() if self.page == self._page_of_value() else UNSET
        moves_away = self._effect != ADD_TRAIN
        clickable = self.editable and self._clickable_page()
        already_there = self._already_there()
        available = self._mover_available()
        building = self.building()
        for cell in self.cells:
            button = command_buttons.button_value(number, cell.row, cell.col)
            items = contents.get(button, ())
            cell.dimmed = cell.blocked = cell.shared = cell.unavailable = False
            cell.ghost = None
            cell.replaced = ()
            cell.reserved = command_buttons.reserved_label(building, button)
            if cell.reserved:
                cell.items = ()
            elif button == chosen and not available:
                cell.items, cell.unavailable = tuple(item for item in items if item != (mover, kind)), True
                cell.ghost = (mover, kind)
            elif button == chosen and not already_there:
                others = tuple(item for item in items if item != (mover, kind))
                rivals = [id_ for id_, item_kind in others if item_kind == kind]
                units = [item for item in others if item[1] == UNIT]
                if kind == TECH and units:
                    # A unit in the slot covers the tech.
                    cell.items, cell.blocked = others, True
                elif kind == TECH:
                    cell.items, cell.shared = ((mover, kind), *others), bool(rivals)
                elif command_buttons.shown_in_slot(mover, rivals, kind):
                    cell.items, cell.replaced = ((mover, kind),), others
                else:
                    cell.items, cell.blocked = others, True
            elif button == chosen:
                cell.items = ((mover, kind),)
            elif moves_away and available and (mover, kind) in items:
                cell.dimmed = True
                cell.items = items
            else:
                cell.items = items
            self._mark(cell, button, button == chosen)
            cell.setChecked(button == chosen)
            cell.setEnabled(clickable and not cell.reserved)
            self._paint_cell(cell)

    def _mark(self, cell: _Cell, button: int, chosen: bool) -> None:
        """Set the cell's frame and the notes on the other effects that changed it."""
        applied = self._applied
        placed = applied.placed.get(button, ()) if applied is not None else ()
        removed = applied.removed.get(button, ()) if applied is not None else ()
        problems = []
        if chosen:
            covers = any(kind == TECH for _id, kind in cell.replaced)
            problems.append(cell.blocked or cell.unavailable or bool(cell.reserved) or covers)
        notes = []
        for edit in placed:
            problem = self._placed_problem(cell.items, edit.item)
            problems.append(bool(problem))
            notes.append(f"{edit.label}: {_name(*edit.item)} here{problem}")
        for edit in applied.unavailable.get(button, ()) if applied is not None else ():
            problems.append(True)
            why = self.context.missing_text(edit.mover, edit.kind, _name(*edit.item))
            notes.append(f"{edit.label}: {_name(*edit.item)} here, but {why}")
            if cell.ghost is None:
                cell.ghost = edit.item
        notes += [f"{edit.label}: {_name(*edit.item)} moved away" for edit in removed]
        cell.notes = tuple(notes)
        cell.frame = "" if not problems else FRAME_PROBLEM if any(problems) else FRAME_OK

    @staticmethod
    def _placed_problem(items, item) -> str:
        """" (hidden)" or " (covers a tech)" for another effect's button in a cell holding `items`, else ""."""
        units = [entry for entry in items if entry[1] == UNIT]
        if item[1] == TECH:
            return " (hidden)" if units else ""
        if not items or items[0] != item:
            return " (hidden)"
        return " (covers a tech)" if len(units) < len(items) else ""

    def _paint_cell(self, cell: _Cell) -> None:
        cell.setIcon(self._cell_icon(cell))
        cell.setText("")
        cell.update()
        if cell.reserved:
            cell.setToolButtonStyle(Qt.ToolButtonTextOnly)
            cell.setText(cell.reserved)
            cell.setToolTip(self._RESERVED_TIP.format(cell.reserved))
            return
        if not cell.items:
            lines = [self._unavailable_tip()] if cell.unavailable else []
            cell.setToolTip("\n".join([*lines, *cell.notes]))
            if cell.ghost is not None and cell.icon().isNull():
                cell.setToolButtonStyle(Qt.ToolButtonTextOnly)
                cell.setText(cell.fontMetrics().elidedText(_name(*cell.ghost), Qt.ElideRight, CELL_PX - 4))
            elif cell.ghost is not None:
                cell.setToolButtonStyle(Qt.ToolButtonIconOnly)
            return
        base_id, base_kind = cell.items[0]
        if cell.icon().isNull():
            cell.setToolButtonStyle(Qt.ToolButtonTextOnly)
            name = _name(base_id, base_kind)
            cell.setText(cell.fontMetrics().elidedText(name, Qt.ElideRight, CELL_PX - 4))
        else:
            cell.setToolButtonStyle(Qt.ToolButtonIconOnly)
        tip = listing(cell.items)
        if cell.dimmed:
            mover = (self.mover(), self.kind)
            rest = listing(tuple(item for item in cell.items if item != mover))
            tip = f"Moved away: {_name(*mover)}\n{rest}".rstrip()
        if cell.shared:
            tip += "\nTechs in one slot take turns: the first one researchable shows, then the next."
        if cell.blocked and self.kind == TECH:
            tip += (
                f"\nHidden here: {_name(self.mover(), self.kind)} (a unit in the slot covers it). "
                f"Move or disable {_name(*cell.items[0])} to free the slot."
            )
        elif cell.blocked:
            tip += (
                f"\nHidden here: {_name(self.mover(), self.kind)} (the lower id shows). "
                f"Move or disable {_name(*cell.items[0])} to free the slot."
            )
        covered = [item for item in cell.replaced if item[1] == TECH]
        if covered:
            names = ", ".join(_name(*item) for item in covered)
            tip += f"\nCovers {names}: can't be researched while this unit is here."
        if cell.unavailable:
            tip += "\n" + self._unavailable_tip()
        if cell.notes:
            tip += "\n" + "\n".join(cell.notes)
        cell.setToolTip(tip)

    def _unavailable_tip(self) -> str:
        mover, kind = self.mover(), self.kind
        tip = f"No button: {self.context.missing_text(mover, kind, _name(mover, kind))}."
        how = train_locations.enable_hint(self.context.missing(mover, kind), kind)
        return f"{tip} Enable it with {how} to show it." if how else tip

    _RESERVED_TIP = "The menu's own {} button. A building put here has no visible button; its hotkey still works."

    def _cell_icon(self, cell: _Cell) -> QIcon:
        if cell.items:
            base, dimmed = self._icon(*cell.items[0]), cell.dimmed
        else:
            base, dimmed = (self._icon(*cell.ghost) if cell.ghost else None), True
        if base is None:
            return QIcon()
        if not dimmed:
            return QIcon(base)
        canvas = QPixmap(ICON_PX, ICON_PX)
        canvas.fill(Qt.transparent)
        painter = QPainter(canvas)
        painter.setOpacity(_DIMMED_OPACITY)
        painter.drawPixmap(0, 0, base)
        painter.end()
        return QIcon(canvas)

    def _icon(self, id_: int, kind: str) -> QPixmap | None:
        """A unit's portrait or a tech's icon from the install, cached per id."""
        if id_ < 0:
            return None
        key = (id_, kind)
        if key not in self._icons:
            if kind == TECH:
                arr = asset_source.get_tech_icon(command_buttons.catalog_row(id_, TECH).get("icon"))
            else:
                arr = asset_source.get_unit_icon(*object_catalog.portrait_for(id_))
            self._icons[key] = None if arr is None else pixmap_from_rgba(arr, ICON_PX)
        return self._icons[key]

    def _footer(self) -> str:
        """Where the unit, tech or building is trained or researched today."""
        lines = []
        homes = command_buttons.home_slots(self.mover(), self.kind)
        if homes:
            page = command_buttons.villager_page(self.mover()) if self.kind == UNIT else None
            places = (f"{building_name(b, page)}, {command_buttons.slot_label(n)}" for b, n in homes)
            lines.append("Now at: " + "; ".join(places))
        return "\n".join(lines)

    # -- input ---------------------------------------------------------------

    def _show_page(self, page: str) -> None:
        self.page = page
        self.refresh()

    def _all_triggers_toggled(self, checked: bool) -> None:
        settings.set_button_grid_all_triggers(checked)
        self.refresh()

    def _cell_clicked(self, cell: _Cell) -> None:
        combo = self._fields.get(SLOT_FIELD)
        if not isinstance(combo, QComboBox) or not self.editable or not self._clickable_page():
            return
        value = command_buttons.button_value(self._page_number(self.page), cell.row, cell.col)
        index = combo.findData(value)
        if index < 0 or index == combo.currentIndex():
            # A click unchecks a checkable button; put the state back.
            self.refresh()
            return
        page = self.page
        combo.setCurrentIndex(index)
        if self._fields:
            # The combo's write may rebuild the form; a surviving grid keeps its page.
            self.page = page
            self.refresh()
