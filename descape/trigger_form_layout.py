"""GH #387: Trigger Details laid out across a wide, short dock above or below the map.

"game" puts each form row in the cell the in-game trigger editor gives its field,
measured from the game's own screens (Create Object, Task Object, Change Object
Armor, Modify Attribute, Display Instructions, Objects in Area and the trigger
row). place() resolves those cells, a row the table does not name taking the
first free one, so every type lays out. "compact" keeps the form's row order and
flow() splits it into columns that fit the dock's height.

place() and flow() are plain functions; arrange_cells() and arrange_columns()
put the rows' existing widgets into a QGridLayout, so their signals stay as built.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from PyQt5.QtWidgets import QFormLayout, QGridLayout, QHBoxLayout, QLabel, QVBoxLayout, QWidget

# Each table entry is (column, row, rowspan, colspan), 1-based like the measurements.
Slot = tuple[int, int, int, int]

GAME_ROWS = 3
TRIGGER_ROWS = 4
# A rowspan that takes every row, for a block too tall to share a column.
FULL_HEIGHT = 0

# Block keys besides spec names: a list type row, a coordinate group's Set / Go to / Reset row,
# the area's four boxes as one grid, and the extra rows. The game's type cell (1, 1) is left
# to the first free row: the panel's Type… button already shows and changes the type.
AREA_GRID = "area_grid"
HINT, FACET_PREVIEW, BUTTON_GRID, INSTRUCTION_PREVIEW = "hint", "facet_preview", "button_grid", "instruction_preview"


def list_type_key(field: str) -> str:
    return f"list_type:{field}"


def group_key(group: str) -> str:
    return f"group:{group}"


def _cell(column: int, row: int, rowspan: int = 1, colspan: int = 1) -> Slot:
    return (column, row, rowspan, colspan)


GAME_SLOTS: dict[str, Slot] = {
    list_type_key("object_list_unit_id"): _cell(1, 2),
    list_type_key("object_list"): _cell(1, 2),
    "object_list_unit_id": _cell(1, 3),
    "object_list": _cell(1, 3),
    "source_player": _cell(2, 1),
    "object_group": _cell(2, 2),
    "object_attributes": _cell(2, 2),
    "instruction_panel_position": _cell(2, 2),
    "facet": _cell(2, 2),
    "object_type": _cell(2, 3),
    "item_id": _cell(2, 3),
    "display_time": _cell(2, 3),
    "selected_object_ids": _cell(3, 1),
    group_key("objects"): _cell(3, 1),
    "string_id": _cell(3, 1),
    AREA_GRID: _cell(3, 2),
    **dict.fromkeys(("area_x1", "area_y1", "area_x2", "area_y2"), _cell(3, 2)),
    group_key("area"): _cell(3, 2),
    # Display Instructions' message box spans two columns and two rows; Modify Attribute's one-line Name does not.
    "message": _cell(3, 2, 2, 2),
    "location_x": _cell(3, 3),
    "location_y": _cell(3, 3),
    group_key("location"): _cell(3, 3),
    "operation": _cell(3, 3),
    "inverted": _cell(3, 3),
    "object_state": _cell(4, 1),
    "sound_name": _cell(4, 1),
    "issue_group_command": _cell(4, 1),
    "quantity": _cell(4, 2),
    "armour_attack_quantity": _cell(4, 2),
    "action_type": _cell(4, 2),
    "armour_attack_class": _cell(4, 3),
    "include_changeable_weapon_objects": _cell(4, 3),
    "disable_garrison_unload_sound": _cell(4, 3),
    "queue_action": _cell(5, 1),
    "play_sound": _cell(5, 1),
    # The game's Object Icon area, under Play Sound and Tag Color: Display Instructions' preview.
    INSTRUCTION_PREVIEW: _cell(5, 2, 1, 2),
    "disable_sound": _cell(6, 1),
    "use_tag_color_for_icon": _cell(6, 1),
}

# A type whose fields read better in another arrangement than the shared table's, by type name.
# Change Variable: its four one-line fields in one row, the operation between name and amount.
TYPE_SLOTS: dict[str, dict[str, Slot]] = {
    "change_variable": {
        "variable": _cell(1, 1),
        "message": _cell(2, 1),
        "operation": _cell(3, 1),
        "quantity": _cell(4, 1),
    },
}

TRIGGER_SLOTS: dict[str, Slot] = {
    "name": _cell(1, 1),
    "enabled": _cell(1, 2),
    "looping": _cell(1, 3),
    "description": _cell(2, 1, 2),
    "display_as_objective": _cell(2, 3),
    "description_string_table_id": _cell(2, 4),
    "short_description": _cell(3, 1, 2),
    "display_on_screen": _cell(3, 3),
    "short_description_string_table_id": _cell(3, 4),
    "header": _cell(4, 1),
    "description_order": _cell(4, 2),
    "mute_objectives": _cell(4, 3),
    "execute_on_load": _cell(4, 4),
}

# Keys that share one cell rather than collide: a group's fields stack under its Set row.
CELL_GROUPS: dict[str, str] = {
    **dict.fromkeys(("location_x", "location_y", group_key("location")), "location"),
    **dict.fromkeys(("area_x1", "area_y1", "area_x2", "area_y2", AREA_GRID, group_key("area")), "area"),
    **dict.fromkeys(("selected_object_ids", group_key("objects")), "objects"),
}

# An unslotted block's (rowspan, colspan) when it needs more than one cell.
FREE_SPANS: dict[str, tuple[int, int]] = {BUTTON_GRID: (FULL_HEIGHT, 1)}


def place(keys: Sequence[str], slots: dict[str, Slot], rows: int) -> dict[str, Slot]:
    """Each key's (column, row, rowspan, colspan).

    A key takes its slot's cell unless an earlier key holds it; keys of one
    CELL_GROUPS group share theirs. A spanning slot spans only if every cell it
    covers is still free once the single cells are placed, so a short form's
    span never pushes a field out of its own cell. A group's cell then grows up
    through the free cells above it, as its boxes make it tall. The rest take the first free
    cell, column by column, past the table's columns if need be. Last, _packed() moves
    every cell up and closes the empty columns."""
    taken: dict[tuple[int, int], str] = {}
    cells: dict[str, Slot] = {}
    group_cell: dict[str, Slot] = {}

    def claim(key: str, cell: Slot) -> None:
        column, row, rowspan, colspan = cell
        for c in range(column, column + colspan):
            for r in range(row, row + rowspan):
                taken[(c, r)] = key
        cells[key] = cell
        if key in CELL_GROUPS:
            group_cell.setdefault(CELL_GROUPS[key], cell)

    for key in keys:
        slot = slots.get(key)
        if slot is None:
            continue
        group = CELL_GROUPS.get(key)
        if group in group_cell and group_cell[group][:2] == slot[:2]:
            cells[key] = group_cell[group]
        elif slot[:2] not in taken:
            claim(key, _cell(slot[0], slot[1]))
    for key in keys:
        if key not in cells or key not in slots or slots[key][2:] == (1, 1):
            continue
        column, row, rowspan, colspan = slots[key]
        covered = [(c, r) for c in range(column, column + colspan) for r in range(row, row + rowspan)]
        if row + rowspan - 1 <= rows and all(taken.get(cell, key) == key for cell in covered):
            claim(key, slots[key])
    for group, (column, row, _rowspan, colspan) in list(group_cell.items()):
        # A group's coordinate boxes make its cell tall, so it starts in any free cell above.
        top = row
        while top > 1 and (column, top - 1) not in taken:
            top -= 1
        if top < row:
            grown = _cell(column, top, row - top + 1, colspan)
            for key, cell in list(cells.items()):
                if cell == group_cell[group]:
                    claim(key, grown)
            group_cell[group] = grown
    for key in keys:
        if key in cells:
            continue
        group = CELL_GROUPS.get(key)
        if group in group_cell:
            cells[key] = group_cell[group]
            continue
        rowspan, colspan = FREE_SPANS.get(key, (1, 1))
        claim(key, _first_free(taken, rows if rowspan == FULL_HEIGHT else min(rowspan, rows), colspan, rows))
    return _packed(cells)


def _packed(cells: dict[str, Slot]) -> dict[str, Slot]:
    """Each cell moved up into the free rows above it, then the empty columns
    closed, so a short form's fields sit together at the top left."""
    moved: dict[Slot, Slot] = {}
    taken: set[tuple[int, int]] = set()
    # Row by row, so every cell that can sit above one has already moved.
    for cell in sorted(set(cells.values()), key=lambda cell: (cell[1], cell[0])):
        column, row, rowspan, colspan = cell
        while row > 1 and all((c, row - 1) not in taken for c in range(column, column + colspan)):
            row -= 1
        taken.update((c, r) for c in range(column, column + colspan) for r in range(row, row + rowspan))
        moved[cell] = _cell(column, row, rowspan, colspan)
    used = sorted({c for column, _r, _rs, colspan in moved.values() for c in range(column, column + colspan)})
    renumber = {column: index + 1 for index, column in enumerate(used)}
    return {key: (renumber[moved[cell][0]], *moved[cell][1:]) for key, cell in cells.items()}


def _first_free(taken: dict[tuple[int, int], str], rowspan: int, colspan: int, rows: int) -> Slot:
    column = 1
    while True:
        for row in range(1, rows - rowspan + 2):
            covered = ((c, r) for c in range(column, column + colspan) for r in range(row, row + rowspan))
            if not any(cell in taken for cell in covered):
                return _cell(column, row, rowspan, colspan)
        column += 1


def flow(heights: Iterable[int], available: int) -> list[int]:
    """Each block's column for Compact: consecutive blocks fill a column up to
    `available`, at least one per column."""
    columns: list[int] = []
    column, used = 0, 0
    for height in heights:
        if used and used + height > available:
            column, used = column + 1, 0
        columns.append(column)
        used += height
    return columns


@dataclass
class Block:
    """One form row as rehomed: its caption (None for a full-width row), its
    widget, and whether the caption sat on its own row above it (a text box)."""

    key: str
    caption: QLabel | None
    field: QWidget
    stacked: bool = False


def _bare_host() -> tuple[QWidget, QVBoxLayout]:
    host = QWidget()
    layout = QVBoxLayout(host)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(2)
    return host, layout


def _caption_above(caption: QLabel) -> QLabel:
    font = caption.font()
    font.setBold(True)
    caption.setFont(font)
    return caption


def arrange_cells(grid: QGridLayout, blocks: Sequence[Block], cells: dict[str, Slot]) -> list[QWidget]:
    """Game layout: one stack per column, its cells in row order, each caption above its
    field. A column stacks on its own, so a tall cell (a coordinate group, a text box)
    never pushes another column's fields down; spans only reserve their cells in place().
    Returns the column hosts."""
    by_column: dict[int, dict[int, list[Block]]] = {}
    for block in blocks:
        column, row, _rowspan, _colspan = cells[block.key]
        by_column.setdefault(column, {}).setdefault(row, []).append(block)
    hosts = []
    for column in sorted(by_column):
        host, layout = _bare_host()
        layout.setSpacing(_CELL_GAP)
        for row in sorted(by_column[column]):
            layout.addLayout(_cell_layout(by_column[column][row]))
        layout.addStretch(1)
        grid.addWidget(host, 0, column - 1)
        hosts.append(host)
    _settle_grid(grid, max(by_column, default=0), 1)
    return hosts


# The gap between two cells of a column, and between a caption and its field.
_CELL_GAP, _CAPTION_GAP = 8, 2
# A coordinate's caption goes beside its box, shortened as the area's X1..Y2 are.
_COORDINATES = frozenset({"location_x", "location_y", "area_x1", "area_y1", "area_x2", "area_y2"})


def _cell_layout(members: list[Block]) -> QVBoxLayout:
    """One cell: a group's Set / Go to / Reset row first, as in the game, then its
    coordinates on one line, each beside its caption; any other block's caption above it."""
    cell = QVBoxLayout()
    cell.setSpacing(_CAPTION_GAP)
    beside = None
    for block in sorted(members, key=lambda b: not b.key.startswith("group:")):
        if block.key in _COORDINATES:
            if beside is None:
                beside = QHBoxLayout()
                cell.addLayout(beside)
            if block.caption is not None:
                beside.addWidget(short_caption(block.key, block.caption))
            beside.addWidget(block.field, 1)
            continue
        if block.caption is not None:
            cell.addWidget(_caption_above(block.caption))
        cell.addWidget(block.field)
    return cell


def short_caption(key: str, caption: QLabel) -> QLabel:
    """A coordinate's caption as its field name's last part (X, Y1), the full caption in its tooltip."""
    if not caption.toolTip():
        caption.setToolTip(caption.text())
    caption.setText(key.rpartition("_")[2].upper())
    return caption


def arrange_columns(grid: QGridLayout, blocks: Sequence[Block], columns: Sequence[int]) -> list[QWidget]:
    """Compact: the blocks in order, one form per column, captions beside their fields. Returns the hosts."""
    hosts: list[QWidget] = []
    forms: list[QFormLayout] = []
    for block, column in zip(blocks, columns, strict=True):
        while len(forms) <= column:
            host = QWidget()
            form = QFormLayout(host)
            form.setContentsMargins(0, 0, 0, 0)
            form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
            grid.addWidget(host, 0, len(forms))
            hosts.append(host)
            forms.append(form)
        form = forms[column]
        if block.caption is not None and block.stacked:
            form.addRow(block.caption)
            form.addRow(block.field)
        elif block.caption is not None:
            form.addRow(block.caption, block.field)
        else:
            form.addRow(block.field)
    _settle_grid(grid, len(forms), 1)
    return hosts


# A cell's narrowest, in average characters: a preview's own minimum is about 1 px, which an
# overflowing grid would give it.
_MIN_CELL_CHARS = 16


def floor_width(host: QWidget) -> None:
    """Keep a cell from being squeezed below 16 characters, or below its own width if it
    asks for less (an empty preview, hidden until it has something to show, asks for none).
    Re-run on every fit, as a preview's wish changes with what it shows."""
    floor = min(host.fontMetrics().averageCharWidth() * _MIN_CELL_CHARS, host.sizeHint().width())
    # An explicit minimum replaces the layout's own, so it never goes below that.
    host.setMinimumWidth(max(floor, host.minimumSizeHint().width(), 0))


def _settle_grid(grid: QGridLayout, columns: int, rows: int) -> None:
    """Columns at their own width with the spare width right of column `columns`, and
    the spare height below row `rows`. A QGridLayout never drops a row or column, so
    every other one is reset: an earlier, wider form's would keep a share of the space."""
    for column in range(max(grid.columnCount(), columns + 1)):
        grid.setColumnStretch(column, 1 if column == columns else 0)
    for row in range(max(grid.rowCount(), rows + 1)):
        grid.setRowStretch(row, 1 if row == rows else 0)


def block_height(block: Block, spacing: int) -> int:
    """A block's height in a Compact column, for flow(): wrapped at its natural width,
    which the grid never squeezes it below."""
    field = _wrapped_height(block.field)
    if block.caption is None:
        return field + spacing
    caption = block.caption.sizeHint().height()
    if block.stacked:
        return caption + field + 2 * spacing
    return max(caption, field) + spacing


def _wrapped_height(widget: QWidget) -> int:
    height = widget.sizeHint().height()
    if widget.hasHeightForWidth():
        height = max(height, widget.heightForWidth(widget.sizeHint().width()))
    return height
