"""GH #169: what a building's command panel holds, for the button grid under
Change Train Location, Add Train Location and Change Research Location.

Qt-free. Reads tools/gen_object_catalog.py's `train` and `research` keys:
`[building, button]` pairs from the .dat's train and research locations. A
button value is 1..15 for the 3x5 grid (row-major) and 21..35 for a second
page (ButtonLocation.dock_page2 is +20); anything else is no button.

Units and techs share one panel (Barracks: units 1..4, techs 6..15), so a
cell can hold either. A cell lists every id the .dat puts there unless the
caller passes `keep(id_, kind)`, the civ filter (GH #230,
civ_availability.keeper()): Castle button 1 holds every civ's unique unit,
and only the Conquistador for Spanish.

Buildings are trained at the builder task unit 118, and their
`interface_kind` picks the villager's economic or military page, so a
building's button value only places it once the page is known.

Measured in game (2026-10-15, Spanish, Imperial): a trigger must name 118
itself, since Villager 83 or 293 moves nothing. Value 0 gives no button.
Change Train Location moves the unit off its old slot. Page 2 works (Militia
at Dock 25). Two units in one slot show the lower id (shown_in_slot()):
Knight 38 over Spearman 93, Spearman 93 over Scout Cavalry 448, and in the
builder menus House 70 over Market 84, Market 84 over Blacksmith 103,
Monastery 104 over University 209, Mill 68 over House 70, Farm 50 over Mining
Camp 584. A hidden unit still leaves its old slot, so it has no button at
all. Techs in one slot take turns instead: the first one researchable shows,
then the next once it is researched (Bloodlines and Squires at Barracks
button 12). The builder menus keep 14 and 15 for their own buttons:
a building put there has no visible button, though its hotkey works.
A unit in a tech's slot covers the tech (2026-10-16: an enabled Mule Cart
at Barracks button 6 left Long Swordsman unresearchable, and Chemistry moved
to Barracks button 1 stayed under the Militia), so units win over techs in a
shared slot (covered_techs(), covering_unit()).
"""

from __future__ import annotations

from collections.abc import Callable

from descape import game_dataset, object_catalog

UNIT = "unit"
TECH = "tech"

ROWS = 3
COLUMNS = 5
PAGE_SIZE = ROWS * COLUMNS
PAGE2_OFFSET = 20

# The villager task unit the .dat trains every building at (VMBLD), and the
# only "trained at" id that moves a villager's build button.
BUILDER_UNIT = 118
# Male and female Villager: a trigger naming either moves nothing.
VILLAGER_IDS = frozenset({83, 293})

# The villager build menus' own buttons: more options, then cancel.
BUILDER_RESERVED = {14: "More", 15: "Cancel"}

ECONOMIC = "economic"
MILITARY = "military"
_PAGE_BY_INTERFACE_KIND = {2: ECONOMIC, 10: MILITARY}


def slot(value: int) -> tuple[int, int, int] | None:
    """(page, row, col), all 1-based, or None for a value with no button."""
    if not isinstance(value, int) or isinstance(value, bool):
        return None
    if 1 <= value <= PAGE_SIZE:
        page, index = 1, value - 1
    elif PAGE2_OFFSET + 1 <= value <= PAGE2_OFFSET + PAGE_SIZE:
        page, index = 2, value - PAGE2_OFFSET - 1
    else:
        return None
    return page, index // COLUMNS + 1, index % COLUMNS + 1


def button_value(page: int, row: int, col: int) -> int:
    """slot()'s inverse."""
    return (page - 1) * PAGE2_OFFSET + (row - 1) * COLUMNS + col


def slot_label(value: int) -> str:
    """"row 1, col 2", "page 2, row 1, col 4", or "no button"."""
    found = slot(value)
    if found is None:
        return "no button"
    page, row, col = found
    cell = f"row {row}, col {col}"
    return cell if page == 1 else f"page 2, {cell}"


def button_values() -> tuple[int, ...]:
    """Every value with a button, page 1 then page 2."""
    return tuple(range(1, PAGE_SIZE + 1)) + tuple(range(PAGE2_OFFSET + 1, PAGE2_OFFSET + PAGE_SIZE + 1))


def is_builder(object_id: int) -> bool:
    return object_id == BUILDER_UNIT


@game_dataset.cache(maxsize=1)
def _rows() -> dict[str, dict[int, dict]]:
    """{UNIT: object rows, TECH: tech rows}. From the committed table alone,
    never an install, so like object_catalog._dat_table() it outlives
    object_catalog.clear_caches()."""
    return {UNIT: object_catalog.dat_objects(), TECH: object_catalog.dat_techs()}


def catalog_row(id_: int, kind: str) -> dict:
    """The committed .dat row for a UNIT or TECH id, or {}. Shared, never mutate it."""
    return _rows()[kind].get(id_, {})



def villager_page(object_id: int) -> str | None:
    """ECONOMIC or MILITARY for a building in a villager's build menu, from
    its .dat interface_kind, or None for anything else."""
    return _PAGE_BY_INTERFACE_KIND.get(catalog_row(object_id, UNIT).get("interface_kind"))


@game_dataset.cache(maxsize=1)
def _index() -> dict[tuple[str, int], dict[int, tuple[int, ...]]]:
    """(kind, building) -> {button: ids}."""
    index: dict[tuple[str, int], dict[int, list[int]]] = {}
    for kind, key in ((UNIT, "train"), (TECH, "research")):
        rows = _rows()[kind]
        for id_, row in rows.items():
            for building, button in row.get(key, ()):
                index.setdefault((kind, building), {}).setdefault(button, []).append(id_)
    return {where: {button: tuple(sorted(ids)) for button, ids in cells.items()} for where, cells in index.items()}


Keep = Callable[[int, str], bool]


def occupants(building: int, kind: str, page: str | None = None, keep: Keep | None = None) -> dict[int, tuple[int, ...]]:
    """{button: ids} for the `kind` (UNIT or TECH) buttons at `building`, or {}.
    The builder unit's are limited to `page` (ECONOMIC or MILITARY) when one
    is given, and every id to those `keep(id_, kind)` accepts when given."""
    if is_builder(building):
        cells = _index().get((kind, BUILDER_UNIT), {})
        if page is not None:
            cells = {button: tuple(i for i in ids if villager_page(i) == page) for button, ids in cells.items()}
    else:
        cells = _index().get((kind, building), {})
    if keep is not None:
        cells = {button: tuple(i for i in ids if keep(i, kind)) for button, ids in cells.items()}
    return {button: ids for button, ids in cells.items() if ids}


def has_buttons(building: int) -> bool:
    """Whether the .dat puts any unit or tech button at `building`."""
    return any((kind, building) in _index() for kind in (UNIT, TECH))


def home_slots(id_: int, kind: str) -> tuple[tuple[int, int], ...]:
    """(building, button) for every place the .dat trains or researches `id_` today."""
    key = "research" if kind == TECH else "train"
    return tuple((building, button) for building, button in catalog_row(id_, kind).get(key, ()))


def _sort_key(id_: int, kind: str) -> tuple[bool, int]:
    """Plain entries first: a unit that is not editor-hidden or a hero, a tech for every civ."""
    row = catalog_row(id_, kind)
    if kind == TECH:
        return (row.get("civ", -1) != -1, id_)
    return (bool(row.get("hidden")) or bool(row.get("hero_mode", 0) & 1), id_)


def slot_values(building: int) -> tuple[int, ...]:
    """The button_location values worth offering for `building`, 0 first:
    1..13 for the builder (no page 2, 14 and 15 are its own), else both pages."""
    if is_builder(building):
        return (0, *(value for value in range(1, PAGE_SIZE + 1) if value not in BUILDER_RESERVED))
    return (0, *button_values())


def reserved_label(building: int, value: int) -> str:
    """"More" or "Cancel" for a slot the builder menus keep, else ""."""
    return BUILDER_RESERVED.get(value, "") if is_builder(building) else ""


def shown_in_slot(mover: int, occupants, kind: str) -> bool:
    """Whether `mover`'s button shows over a slot's other `occupants`: the
    lower id wins for units and buildings, among those that are not editor-
    hidden or heroes (base_of(); economic slot 7's hidden BLAC3 18 does not
    beat Market 84, measured). A tech always gets its turn: techs
    in one slot show one after another, as each becomes researchable."""
    if kind == TECH:
        return True
    blocker = base_of(occupants, kind)
    return blocker is None or mover < blocker


def blocker(mover: int, building: int, value: int, keep: Keep | None = None) -> int | None:
    """The unit whose lower id hides `mover`'s button at (building, value),
    or None when it shows, the slot is free, mover already sits there or its
    builder menu is unknown. Only occupants `keep` accepts count."""
    if (BUILDER_UNIT if is_builder(building) else building, value) in home_slots(mover, UNIT):
        return None
    page = villager_page(mover) if is_builder(building) else None
    if is_builder(building) and page is None:
        # Its interface_kind names neither menu, so there is no telling which slot it lands in.
        return None
    rivals = [id_ for id_ in occupants(building, UNIT, page, keep).get(value, ()) if id_ != mover]
    if shown_in_slot(mover, rivals, UNIT):
        return None
    return base_of(rivals, UNIT)


def covered_techs(building: int, value: int, keep: Keep | None = None) -> tuple[int, ...]:
    """The techs a unit put at (building, value) covers: every one `keep`
    accepts there. The builder menus hold no techs."""
    if is_builder(building):
        return ()
    return ordered(occupants(building, TECH, keep=keep).get(value, ()), TECH)


def covering_unit(tech: int, building: int, value: int, keep: Keep | None = None) -> int | None:
    """The unit that hides `tech` put at (building, value), or None when no
    unit `keep` accepts sits there or the tech already does."""
    if is_builder(building) or (building, value) in home_slots(tech, TECH):
        return None
    return base_of(occupants(building, UNIT, keep=keep).get(value, ()), UNIT)


def building_name(building: int, page: str | None = None) -> str:
    """The builder unit reads as the villager's build menu, with the page when known."""
    if is_builder(building):
        return f"Villager build menu ({page})" if page else "Villager build menu"
    return object_catalog.display_name(building)


def ordered(ids, kind: str) -> tuple[int, ...]:
    """`ids` with the cell's base entry first, heroes, hidden and civ-only ones last."""
    return tuple(sorted(ids, key=lambda id_: _sort_key(id_, kind)))


def base_of(ids, kind: str) -> int | None:
    """The entry a cell's icon shows (Militia for Barracks button 1), or None for no ids."""
    found = ordered(ids, kind)
    return found[0] if found else None
