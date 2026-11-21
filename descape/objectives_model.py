"""GH #391: the scenario's objectives in game order, and Description Order
rewrites for the Objectives panel. Qt-free.

A row is any trigger shown as an objective, a header or an on-screen line
(trigger_status._OBJECTIVE_FLAGS). The game lists higher Description Order
first; how it breaks a tie is unconfirmed, so ties keep display order.

Every rewrite returns only the orders that change, and stays within the
field's u32 range: a moved or new row takes a value strictly between its new
neighbours when there is room; else it takes the value under the row above and
the rows below shift down one each until a gap; only when that would go below
0 is the whole list renumbered.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from itertools import pairwise
from typing import Any, NamedTuple

from descape.trigger_status import _OBJECTIVE_FLAGS, read_value

HEADER = "header"
OBJECTIVE = "objective"
# The key insert_order() gives the row it makes room for.
NEW = -1
MAX_ORDER = 2**32 - 1
# The game's "no string" values for a string table id.
_NO_STRING_IDS = (0, -1)


class ObjectiveRow(NamedTuple):
    index: int
    trigger_id: int
    name: str
    order: int
    header: bool
    objective: bool
    on_screen: bool
    mute: bool
    enabled: bool
    short: str
    short_stid: int


class BoxLine(NamedTuple):
    """One line of the in-game box: a HEADER (no checkbox) or an OBJECTIVE."""

    kind: str
    row: ObjectiveRow


def _int(value: Any, default: int = 0) -> int:
    return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else default


def has_string_id(stid: int) -> bool:
    return stid not in _NO_STRING_IDS


def objective_rows(manager) -> list[ObjectiveRow]:
    """Every objective, header or on-screen trigger in game order: higher
    Description Order first, then display order."""
    triggers = list(manager.triggers)
    slot = {index: position for position, index in enumerate(manager.trigger_display_order)}
    rows = []
    for index, trigger in enumerate(triggers):
        if not any(read_value(trigger, flag) for flag in _OBJECTIVE_FLAGS):
            continue
        rows.append(
            ObjectiveRow(
                index=index,
                trigger_id=_int(read_value(trigger, "trigger_id"), index),
                name=read_value(trigger, "name") or "",
                order=_int(read_value(trigger, "description_order")),
                header=bool(read_value(trigger, "header")),
                objective=bool(read_value(trigger, "display_as_objective")),
                on_screen=bool(read_value(trigger, "display_on_screen")),
                mute=bool(read_value(trigger, "mute_objectives")),
                enabled=bool(read_value(trigger, "enabled")),
                short=read_value(trigger, "short_description") or "",
                short_stid=_int(read_value(trigger, "short_description_stid")),
            )
        )
    rows.sort(key=lambda row: (-row.order, slot.get(row.index, len(slot) + row.index)))
    return rows


def box_lines(rows: Iterable[ObjectiveRow], at_start: bool) -> list[BoxLine]:
    """What the game's box draws: every row with Display on Screen set, minus
    triggers disabled at game start when `at_start`."""
    return [
        BoxLine(HEADER if row.header else OBJECTIVE, row)
        for row in rows
        if row.on_screen and (row.enabled or not at_start)
    ]


def _spread(above: int | None, below: int | None, count: int) -> list[int] | None:
    """`count` distinct descending orders strictly between `above` and
    `below` (None: no neighbour that side), or None when there is no room."""
    if above is None and below is None:
        return None
    if above is None:
        return [below + count - i for i in range(count)] if below + count <= MAX_ORDER else None
    low = -1 if below is None else below
    if above - low - 1 < count:
        return None
    return [low + (above - low) * (count - i) // (count + 1) for i in range(count)]


def _fit(keys: Sequence[int], orders: dict[int, int], placed: Sequence[int]) -> dict[int, int]:
    """New orders for `keys` (the list as it should read, top first) where
    only the contiguous `placed` block is out of place; `orders` holds every
    other key's current order. Only changed values are returned."""
    at = keys.index(placed[0])
    above = orders[keys[at - 1]] if at > 0 else None
    below = orders[keys[at + len(placed)]] if at + len(placed) < len(keys) else None
    current = [orders.get(key) for key in placed]
    if None not in current and all(
        (above is None or above > value) and (below is None or value > below) for value in current
    ) and all(a > b for a, b in pairwise(current)):
        return {}
    values = _spread(above, below, len(placed))
    if values is not None:
        new = dict(zip(placed, values, strict=True))
    elif above is not None and (shifted := _shift_down(keys[at:], orders, set(placed), above)) is not None:
        new = shifted
    else:
        top = max([len(keys) - 1, *(value for key, value in orders.items() if key in keys)])
        new = {key: top - position for position, key in enumerate(keys)}
    return {key: value for key, value in new.items() if orders.get(key) != value}


def _shift_down(keys: Sequence[int], orders: dict[int, int], placed: set[int], above: int) -> dict[int, int] | None:
    """Each of `keys` (the placed block first) one under the last, starting
    under `above`, until a row already sits lower; None if that goes below 0."""
    new = {}
    value = above
    for key in keys:
        current = orders.get(key)
        if key not in placed and current is not None and current < value:
            break
        value -= 1
        if value < 0:
            return None
        new[key] = value
    return new


def orders_after_move(rows: Sequence[ObjectiveRow], moved: Iterable[int], to_position: int) -> dict[int, int]:
    """Orders that put the rows with trigger indices `moved` (kept in their
    list order) at `to_position` of the list without them, as one block."""
    moved_set = set(moved)
    block = [row.index for row in rows if row.index in moved_set]
    if not block:
        return {}
    rest = [row.index for row in rows if row.index not in moved_set]
    to_position = max(0, min(to_position, len(rest)))
    keys = rest[:to_position] + block + rest[to_position:]
    if keys == [row.index for row in rows]:
        return {}
    return _fit(keys, {row.index: row.order for row in rows}, block)


def renumbered(rows: Sequence[ObjectiveRow], selected: Iterable[int] = ()) -> dict[int, int]:
    """The Renumber button: counting down by 1 in list order, the top row
    keeping its value (raised so the last row lands on 0 or more).

    With two or more rows `selected` (trigger indices), only they count down,
    from the first one's value, each one under the row before it; a row
    between or below them moves down only where it would otherwise collide,
    so the list order never changes. Below 0, the whole list is renumbered."""
    if not rows:
        return {}
    wanted = {row.index for row in rows} & set(selected)
    if len(wanted) >= 2:
        positions = [position for position, row in enumerate(rows) if row.index in wanted]
        new: dict[int, int] = {}
        last = rows[positions[0]].order
        for position in range(positions[0] + 1, len(rows)):
            row = rows[position]
            if row.index not in wanted and row.order < last:
                if position > positions[-1]:
                    break
                last = row.order
                continue
            last -= 1
            if last < 0:
                return renumbered(rows)
            if row.order != last:
                new[row.index] = last
        return new
    top = max(rows[0].order, len(rows) - 1)
    new = {row.index: top - position for position, row in enumerate(rows)}
    return {row.index: new[row.index] for row in rows if row.order != new[row.index]}


def insert_order(rows: Sequence[ObjectiveRow], after_index: int | None) -> dict[int, int]:
    """Orders for a new row (key NEW) placed under the row with trigger index
    `after_index`, or at the bottom when it is None or not a row. Other keys
    are rows renumbered to make room."""
    keys = [row.index for row in rows]
    at = keys.index(after_index) + 1 if after_index in keys else len(keys)
    if not keys:
        return {NEW: 0}
    return _fit([*keys[:at], NEW, *keys[at:]], {row.index: row.order for row in rows}, [NEW])


def moved_by_step(rows: Sequence[ObjectiveRow], indices: Iterable[int], delta: int) -> dict[int, int]:
    """The up and down buttons: the selected rows' block one row up (-1) or
    down (+1) from its first row. {} at an edge."""
    wanted = set(indices)
    positions = [position for position, row in enumerate(rows) if row.index in wanted]
    if not positions:
        return {}
    rest_position = positions[0] + delta
    if rest_position < 0 or (delta > 0 and positions[-1] + 1 >= len(rows)):
        return {}
    return orders_after_move(rows, wanted, rest_position)


def insert_orders(rows: Sequence[ObjectiveRow], after_index: int | None, new_keys: Sequence[int]) -> dict[int, int]:
    """insert_order() for several new rows (trigger indices `new_keys`, not yet
    rows) placed under `after_index` in the given order. Rows renumbered to
    make room are included."""
    rows = list(rows)
    result: dict[int, int] = {}
    anchor = after_index
    for key in new_keys:
        changes = insert_order(rows, anchor)
        value = changes.pop(NEW)
        result.update(changes)
        rows = [row._replace(order=changes.get(row.index, row.order)) for row in rows]
        at = next((position + 1 for position, row in enumerate(rows) if row.index == anchor), len(rows))
        rows.insert(at, ObjectiveRow(key, key, "", value, False, True, True, False, True, "", 0))
        result[key] = value
        anchor = key
    return result


# The game draws a short description's leading "+" as a square checkbox and "-" as a round
# (optional objective) one, never as text (user's in-game screenshots, 2026-11-21).
REQUIRED = "required"
OPTIONAL = "optional"
_MARKERS = {"+": REQUIRED, "-": OPTIONAL}


def split_marker(text: str, tag_names: Iterable[str] = ()) -> tuple[str | None, str]:
    """(REQUIRED, OPTIONAL or None, the text without its marker). Colour tags
    (`tag_names`, any case) before the marker stay; spaces after it go."""
    tags = {name.upper() for name in tag_names}
    position = 0
    while position < len(text):
        if text[position].isspace():
            position += 1
            continue
        end = text.find(">", position)
        if text[position] == "<" and end > 0 and text[position + 1 : end].upper() in tags:
            position = end + 1
            continue
        break
    if position < len(text) and text[position] in _MARKERS:
        return _MARKERS[text[position]], text[:position].strip() + text[position + 1 :].lstrip()
    return None, text
